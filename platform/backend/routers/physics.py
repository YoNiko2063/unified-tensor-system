"""
POST /api/v1/physics/simulate  body: {system_type, params, n_steps}

system_type: "rlc" | "harmonic" | "duffing"
"""
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SimulateRequest(BaseModel):
    system_type: str
    params: Dict[str, Any] = {}
    n_steps: int = 400


class TrajectoryPoint(BaseModel):
    t: float
    x: float
    v: float


class SimulateResponse(BaseModel):
    trajectory: List[TrajectoryPoint]
    koopman_trust: float
    regime_type: str
    omega0: float
    Q: float
    system_type: str


def _simulate_rlc(params: Dict[str, Any], n_steps: int) -> SimulateResponse:
    from optimization.rlc_evaluator import RLCEvaluator, RLCParams

    R = float(params.get("R", 10.0))
    L = float(params.get("L", 0.01))
    C = float(params.get("C", 1e-6))
    target_hz = float(params.get("target_hz", 1000.0))

    rlc = RLCParams(R=R, L=L, C=C)
    evaluator = RLCEvaluator()
    result = evaluator.evaluate(rlc, target_hz)

    omega0 = float(evaluator.cutoff_frequency_rad(rlc))
    Q = float(evaluator.Q_factor(rlc))

    # RK4 trajectory: state = [charge q, current i=dq/dt]
    # L·i' + R·i + q/C = 0  =>  i' = -(R/L)·i - q/(L·C)
    dt = 0.001
    x_arr = np.array([0.0, 1e-3])  # [q0, i0]
    traj: List[TrajectoryPoint] = []

    for step in range(min(n_steps, 600)):
        t = step * dt
        traj.append(TrajectoryPoint(t=t, x=float(x_arr[0]), v=float(x_arr[1])))

        def rhs(s):
            q, i = s
            return np.array([i, -(R / L) * i - q / (L * C)])

        k1 = rhs(x_arr)
        k2 = rhs(x_arr + 0.5 * dt * k1)
        k3 = rhs(x_arr + 0.5 * dt * k2)
        k4 = rhs(x_arr + dt * k3)
        x_arr = x_arr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    regime = "lca" if Q >= 0.5 else "overdamped"

    return SimulateResponse(
        trajectory=traj,
        koopman_trust=0.85 if result.constraints_ok else 0.40,
        regime_type=regime,
        omega0=omega0,
        Q=Q,
        system_type="rlc",
    )


def _simulate_harmonic(params: Dict[str, Any], n_steps: int) -> SimulateResponse:
    """Simple harmonic oscillator: x'' + 2ζω₀x' + ω₀²x = 0"""
    omega0 = float(params.get("omega0", 6.283))
    zeta = float(params.get("zeta", 0.1))
    x0 = float(params.get("x0", 1.0))
    v0 = float(params.get("v0", 0.0))

    Q = omega0 / (2 * zeta * omega0 + 1e-12)
    dt = float(params.get("dt", 0.01))
    traj: List[TrajectoryPoint] = []
    state = np.array([x0, v0])

    for step in range(min(n_steps, 600)):
        t = step * dt
        traj.append(TrajectoryPoint(t=t, x=float(state[0]), v=float(state[1])))

        def rhs(s):
            x, v = s
            return np.array([v, -(omega0 ** 2) * x - 2 * zeta * omega0 * v])

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    koopman_trust = min(0.95, 0.5 + 0.5 * (1.0 - zeta))
    regime = "lca" if zeta < 0.5 else "overdamped"

    return SimulateResponse(
        trajectory=traj,
        koopman_trust=koopman_trust,
        regime_type=regime,
        omega0=omega0,
        Q=Q,
        system_type="harmonic",
    )


def _simulate_duffing(params: Dict[str, Any], n_steps: int) -> SimulateResponse:
    from optimization.duffing_evaluator import DuffingEvaluator, DuffingParams

    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 0.1))
    delta = float(params.get("delta", 0.3))
    x0 = float(params.get("x0", 1.0))
    v0 = float(params.get("v0", 0.0))

    dparams = DuffingParams(alpha=alpha, beta=beta, delta=delta)
    evaluator = DuffingEvaluator(dparams, n_steps=min(n_steps, 600))
    result = evaluator.evaluate(x0=x0, v0=v0)

    # Build trajectory from raw RK4 integration (re-run for cleanliness)
    dt = 0.05
    state = np.array([x0, v0])
    traj: List[TrajectoryPoint] = []

    for step in range(min(n_steps, 600)):
        t = step * dt
        traj.append(TrajectoryPoint(t=t, x=float(state[0]), v=float(state[1])))

        def rhs(s, _alpha=alpha, _beta=beta, _delta=delta):
            x, v = s
            return np.array([v, -_delta * v - _alpha * x - _beta * x ** 3])

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    kr = result.koopman_result
    trust = float(kr.koopman_trust) if kr else 0.0
    regime = "abelian" if result.is_linear_regime else ("near_separatrix" if result.near_separatrix else "nonabelian")

    return SimulateResponse(
        trajectory=traj,
        koopman_trust=trust,
        regime_type=regime,
        omega0=float(result.omega0_eff),
        Q=float(result.Q_linear),
        system_type="duffing",
    )


_SIMULATORS = {
    "rlc": _simulate_rlc,
    "harmonic": _simulate_harmonic,
    "duffing": _simulate_duffing,
}


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    """Run a physics simulation and return the trajectory + Koopman metadata."""
    if req.system_type not in _SIMULATORS:
        raise HTTPException(
            status_code=422,
            detail=f"system_type must be one of {list(_SIMULATORS.keys())}, got {req.system_type!r}",
        )
    if req.n_steps < 1 or req.n_steps > 2000:
        raise HTTPException(status_code=422, detail="n_steps must be between 1 and 2000")

    try:
        return _SIMULATORS[req.system_type](req.params, req.n_steps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc
