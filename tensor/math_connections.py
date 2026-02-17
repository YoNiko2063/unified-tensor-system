"""Math connections: bridges ECEMath math modules to the tensor improvement loop.

Seven connections:
  1. Fisher → GSD planning (FIM eigenvalues guide improvement directions)
  2. Regime detection → run_system monitoring (pause on L2 regime shift)
  3. Stochastic solver → explorer robustness (20-path check on high scorers)
  4. Neural ODE prediction error → GSD weights (high error = improve first)
  5. SNN free energy firing → L1 activation (free energy threshold activation)
  6. Ground truth (pytest) → jump events (test results as L2 discontinuities)
  7. Feed health → run_system monitoring (L0 staleness + status)
"""
import os
import sys
import time
import subprocess
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ECEMATH_SRC = os.path.join(_ROOT, 'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.matrix import MNASystem
from core.fisher import FisherInformation, FisherResult
from core.regime import RegimeSwitchingSystem, RegimeResult
from core.stochastic import StochasticDynamicalSystem, DiffusionTerm, StochasticResult
from core.dynamical_system import DynamicalSystem
from core.sparse_solver import compute_free_energy, compute_harmonic_signature
from core.coarsening import CoarseGrainingOperator
from tensor.core import UnifiedTensor


# ═══════════════════════════════════════════════════════════
# 1. FISHER → GSD PLANNING
# ═══════════════════════════════════════════════════════════

@dataclass
class FisherGuidance:
    """FIM-guided planning output."""
    high_info_directions: np.ndarray  # Top eigenvectors (most informative)
    eigenvalues: np.ndarray
    priority_indices: List[int]  # Node indices sorted by information content
    condition_number: float


def fisher_guided_planning(tensor: UnifiedTensor, level: int = 2,
                           top_k: int = 5) -> FisherGuidance:
    """Compute FIM for tensor level, return planning guidance.

    High FIM eigenvalue directions = where small parameter changes cause
    large state changes = highest-priority improvement directions.
    """
    mna = tensor._mna.get(level)
    if mna is None:
        return FisherGuidance(
            high_info_directions=np.zeros((1, 1)),
            eigenvalues=np.array([1.0]),
            priority_indices=[0],
            condition_number=1.0,
        )

    G = mna.G
    n = mna.n_total

    # Normalize G so FIM eigenvalues are O(1), not O(scale^{-4}).
    # Without this, 1e-6 diagonal loading dominates and FIM is uniform.
    # Clamp diagonal to min 1% of mean to prevent near-zero self-conductance
    # nodes from producing 1/ε² blowup in the FIM.
    scale = np.abs(G.diagonal()).mean()
    if scale > 1e-12:
        G_work = G / scale
        diag_min = 0.01
        d = G_work.diagonal().copy()
        d[d < diag_min] = diag_min
        np.fill_diagonal(G_work, d)
    else:
        G_work = G.copy()

    # Sensitivity function: how state changes with G perturbations
    # J_ij = ∂v_i/∂G_jj (diagonal sensitivity)
    def sensitivity_func(theta):
        # theta = diagonal of G_work; J = -G^{-1} (from G·v = u, dv/dG = -G^{-1})
        G_pert = G_work.copy()
        np.fill_diagonal(G_pert, theta)
        # Use truncated SVD pseudoinverse to suppress near-singular modes.
        # This keeps FIM eigenvalues bounded even when G has rank-deficient blocks.
        U, s, Vt = np.linalg.svd(G_pert)
        s_inv = np.where(s > 1e-3, 1.0 / s, 0.0)
        G_inv = (Vt.T * s_inv) @ U.T
        return -G_inv  # (n_states, n_params)

    theta = np.diag(G_work)
    fisher = FisherInformation(sensitivity_func, n_params=n)
    result = fisher.compute(theta)

    # Priority: nodes with highest FIM diagonal (per-node information content).
    # FIM_ii = how much information node i contributes to the total Fisher info.
    top_k = min(top_k, n)
    fim_diag = np.diag(result.fisher_matrix)
    priority = np.argsort(-fim_diag)[:top_k]

    cond = result.eigenvalues[0] / max(result.eigenvalues[-1], 1e-30)

    return FisherGuidance(
        high_info_directions=result.eigenvectors[:, :top_k],
        eigenvalues=result.eigenvalues,
        priority_indices=priority.tolist(),
        condition_number=float(cond),
    )


# ═══════════════════════════════════════════════════════════
# 2. REGIME DETECTION → MONITORING
# ═══════════════════════════════════════════════════════════

@dataclass
class RegimeStatus:
    """Current regime detection result."""
    current_regime: int
    n_regimes: int
    regime_duration: float
    transition_probability: float
    should_pause: bool


def detect_regime(tensor: UnifiedTensor, level: int = 2,
                  history_window: int = 5) -> RegimeStatus:
    """Detect regime at tensor level from eigenvalue gap history.

    If eigenvalue gap narrows significantly → regime transition imminent → pause GSD.
    """
    mna = tensor._mna.get(level)
    if mna is None:
        return RegimeStatus(0, 1, float('inf'), 0.0, False)

    G = mna.G
    n = mna.n_total

    eigvals = np.sort(np.abs(np.linalg.eigvalsh(G)))[::-1]
    if len(eigvals) < 2:
        return RegimeStatus(0, 1, float('inf'), 0.0, False)

    # Regime classification from eigenvalue gap
    gap = (eigvals[0] - eigvals[1]) / max(eigvals[0], 1e-30)

    if gap > 0.5:
        regime = 0  # Stable
    elif gap > 0.2:
        regime = 1  # Transitioning
    else:
        regime = 2  # Bifurcating

    # Transition probability from gap dynamics
    # Narrow gap = high transition probability
    transition_prob = float(np.exp(-5.0 * gap))

    # Should pause if transitioning or bifurcating
    should_pause = regime >= 1 and transition_prob > 0.3

    # Estimate regime duration from eigenvalue gap
    duration = 1.0 / max(transition_prob, 1e-6)

    return RegimeStatus(
        current_regime=regime,
        n_regimes=3,
        regime_duration=duration,
        transition_probability=transition_prob,
        should_pause=should_pause,
    )


# ═══════════════════════════════════════════════════════════
# 3. STOCHASTIC → EXPLORER ROBUSTNESS
# ═══════════════════════════════════════════════════════════

@dataclass
class RobustnessResult:
    """Result of stochastic robustness check."""
    mean_score: float
    std_score: float
    min_score: float
    robust: bool  # std/mean < threshold
    n_paths: int


def stochastic_robustness_check(G: np.ndarray, score_fn, n_paths: int = 20,
                                noise_level: float = 0.05,
                                seed: int = 42) -> RobustnessResult:
    """Run n_paths stochastic perturbations of G, score each.

    A configuration is robust if its score doesn't vary much under noise.
    Uses Euler-Maruyama-style perturbation (not full SDE, just noise injection).
    """
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    scores = []

    for _ in range(n_paths):
        # Perturb G symmetrically
        noise = rng.standard_normal((n, n)) * noise_level
        noise = 0.5 * (noise + noise.T)
        np.fill_diagonal(noise, 0)
        G_pert = G + noise
        # Re-enforce diagonal dominance
        for i in range(n):
            off = G_pert[i, :].copy()
            off[i] = 0
            G_pert[i, i] = -off.sum() + 1e-6

        eigvals = np.sort(np.abs(np.linalg.eigvalsh(G_pert)))[::-1]
        from core.sparse_solver import consonance_score_from_ratios, nearest_consonant
        clean = eigvals[eigvals > 1e-10]
        if len(clean) < 2:
            scores.append(0.0)
            continue
        ratios = clean / clean[0]
        cons = float(consonance_score_from_ratios(ratios))
        if abs(ratios[1]) > 1e-30:
            dom, _ = nearest_consonant(abs(ratios[0] / ratios[1]))
        else:
            dom = 'unison'
        scores.append(float(score_fn(clean, cons, dom)))

    scores = np.array(scores)
    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    min_s = float(np.min(scores))
    robust = std_s / max(mean_s, 1e-30) < 0.3  # CV < 30%

    return RobustnessResult(
        mean_score=mean_s,
        std_score=std_s,
        min_score=min_s,
        robust=robust,
        n_paths=n_paths,
    )


# ═══════════════════════════════════════════════════════════
# 4. NEURAL ODE PREDICTION ERROR → GSD WEIGHTS
# ═══════════════════════════════════════════════════════════

@dataclass
class PredictionErrorReport:
    """Neural ODE prediction error per module."""
    errors: Dict[str, float]  # module_name → prediction error
    mean_error: float
    high_error_modules: List[str]  # Modules needing improvement


def neural_prediction_error(tensor: UnifiedTensor,
                            actual_state: np.ndarray,
                            level: int = 1) -> PredictionErrorReport:
    """Compare neural bridge predicted L1 state vs actual observed state.

    High prediction error for a node = the model is wrong there = improve first.
    Returns per-node errors that can weight GSD task priority.
    """
    predicted = tensor.get_state(level)
    if predicted is None:
        predicted = np.zeros_like(actual_state)

    n = min(len(predicted), len(actual_state))
    errors_vec = np.abs(predicted[:n] - actual_state[:n])

    # Map to module names (use node indices as names if no code graph)
    errors_dict = {}
    for i in range(n):
        errors_dict[f'node_{i}'] = float(errors_vec[i])

    mean_err = float(np.mean(errors_vec))

    # High error = above mean + 1 std
    threshold = mean_err + float(np.std(errors_vec))
    high_error = [name for name, e in errors_dict.items() if e > threshold]

    return PredictionErrorReport(
        errors=errors_dict,
        mean_error=mean_err,
        high_error_modules=high_error,
    )


# ═══════════════════════════════════════════════════════════
# 5. SNN FREE ENERGY FIRING → L1 ACTIVATION
# ═══════════════════════════════════════════════════════════

@dataclass
class FiringActivation:
    """Free energy firing result for L1."""
    firing_mask: np.ndarray
    free_energies: np.ndarray
    n_firing: int
    activation_vector: np.ndarray  # Sparse: 1.0 where firing, 0.0 otherwise


def snn_firing_activation(tensor: UnifiedTensor,
                          tau: float = 1.0, gamma: float = 0.5,
                          theta_base: float = 0.0) -> FiringActivation:
    """Apply free energy firing rule to L1 neural layer.

    F(node_i) = E(node_i) - tau * S(node_i) + gamma * H(node_i)
    Nodes fire when F < theta.
    Returns activation vector for selective L1 update.
    """
    mna = tensor._mna.get(1)
    if mna is None:
        return FiringActivation(
            firing_mask=np.array([False]),
            free_energies=np.array([0.0]),
            n_firing=0,
            activation_vector=np.array([0.0]),
        )

    x = tensor.get_state(1)
    if x is None:
        x = np.zeros(mna.n_total)

    n = mna.n_total
    if n < 3:
        return FiringActivation(
            firing_mask=np.ones(n, dtype=bool),
            free_energies=np.zeros(n),
            n_firing=n,
            activation_vector=np.ones(n),
        )

    k = max(1, min(n - 1, int(np.sqrt(n))))
    try:
        phi = CoarseGrainingOperator(mna, k=k, tolerance=1.0)
        phi.coarsen()
    except ValueError:
        return FiringActivation(
            firing_mask=np.ones(n, dtype=bool),
            free_energies=np.zeros(n),
            n_firing=n,
            activation_vector=np.ones(n),
        )

    firing = compute_free_energy(mna, x, phi, tau=tau, gamma=gamma,
                                  theta_base=theta_base)

    activation = np.where(firing.firing_mask, 1.0, 0.0)

    return FiringActivation(
        firing_mask=firing.firing_mask,
        free_energies=firing.free_energies,
        n_firing=firing.n_firing,
        activation_vector=activation,
    )


# ═══════════════════════════════════════════════════════════
# 6. GROUND TRUTH (PYTEST) → JUMP EVENTS
# ═══════════════════════════════════════════════════════════

@dataclass
class TestJumpEvent:
    """Result of running tests as a jump event for the tensor."""
    tests_passed: int
    tests_failed: int
    tests_total: int
    success_rate: float
    is_jump: bool  # True if significant change from baseline
    jump_magnitude: float  # How much the test results changed


def ground_truth_pytest(test_dir: str = 'tests',
                        baseline_pass_rate: float = 1.0) -> TestJumpEvent:
    """Run pytest and convert results to a tensor jump event.

    A 'jump' occurs when test pass rate changes significantly from baseline.
    This feeds into the stochastic model as a Poisson jump in L2.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_dir, '-q', '--tb=no'],
            capture_output=True, text=True, timeout=120,
            cwd=_ROOT,
        )
        output = result.stdout + result.stderr

        # Parse pytest output: "X passed, Y failed" or "X passed"
        passed = 0
        failed = 0
        for line in output.split('\n'):
            line = line.strip()
            if 'passed' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'passed' and i > 0:
                        try:
                            passed = int(parts[i - 1])
                        except ValueError:
                            pass
                    if p == 'failed' and i > 0:
                        try:
                            failed = int(parts[i - 1])
                        except ValueError:
                            pass

        total = passed + failed
        rate = passed / max(total, 1)
        jump_mag = abs(rate - baseline_pass_rate)
        is_jump = jump_mag > 0.1  # >10% change is a jump

        return TestJumpEvent(
            tests_passed=passed,
            tests_failed=failed,
            tests_total=total,
            success_rate=rate,
            is_jump=is_jump,
            jump_magnitude=jump_mag,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return TestJumpEvent(
            tests_passed=0, tests_failed=0, tests_total=0,
            success_rate=0.0, is_jump=True, jump_magnitude=1.0,
        )


# ═══════════════════════════════════════════════════════════
# 7. FEED HEALTH → MONITORING
# ═══════════════════════════════════════════════════════════

@dataclass
class FeedHealth:
    """Health status of the L0 data feed."""
    is_healthy: bool
    staleness_seconds: float
    ticks_per_minute: float
    l0_populated: bool
    regime: str
    warnings: List[str]


def check_feed_health(tensor: UnifiedTensor,
                      feed_status: Optional[dict] = None,
                      max_staleness: float = 60.0) -> FeedHealth:
    """Check L0 feed health from tensor state and feed status.

    Reports staleness, tick rate, and any warnings.
    """
    warnings = []

    # Check L0 populated
    mna = tensor._mna.get(0)
    l0_pop = mna is not None

    if not l0_pop:
        warnings.append("L0 not populated — no market data")

    # Feed status checks
    staleness = float('inf')
    ticks_pm = 0.0
    regime = 'unknown'

    if feed_status is not None:
        last_update = feed_status.get('last_update', 0.0)
        if last_update > 0:
            staleness = time.time() - last_update
        else:
            staleness = float('inf')

        ticks = feed_status.get('ticks_received', 0)
        running = feed_status.get('running', False)
        regime = feed_status.get('current_regime', 'unknown')

        if not running:
            warnings.append("Feed not running")
        if ticks == 0:
            warnings.append("No ticks received")
        if staleness > max_staleness:
            warnings.append(f"Feed stale: {staleness:.1f}s since last update")

        # Estimate ticks per minute
        if last_update > 0 and ticks > 0:
            elapsed = time.time() - (last_update - ticks * 5.0)  # Rough estimate
            ticks_pm = ticks / max(elapsed / 60.0, 0.01)
    else:
        warnings.append("No feed status available")

    is_healthy = len(warnings) == 0 and staleness < max_staleness

    return FeedHealth(
        is_healthy=is_healthy,
        staleness_seconds=staleness,
        ticks_per_minute=ticks_pm,
        l0_populated=l0_pop,
        regime=regime,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════
# UNIFIED CLASS INTERFACE
# ═══════════════════════════════════════════════════════════

class MathConnections:
    """Unified interface to all math→tensor connections."""

    def __init__(self, tensor: 'UnifiedTensor'):
        self.tensor = tensor

    def fisher_guided_planning(self, level: int = 2, top_k: int = 5) -> FisherGuidance:
        return fisher_guided_planning(self.tensor, level=level, top_k=top_k)

    def detect_regime(self, level: int = 2, history_window: int = 5) -> RegimeStatus:
        return detect_regime(self.tensor, level=level, history_window=history_window)

    @staticmethod
    def stochastic_robustness_check(G: np.ndarray, score_fn, n_paths: int = 20,
                                    noise_level: float = 0.05,
                                    seed: int = 42) -> RobustnessResult:
        return stochastic_robustness_check(G, score_fn, n_paths=n_paths,
                                           noise_level=noise_level, seed=seed)

    def neural_prediction_error(self, actual_state: np.ndarray,
                                level: int = 1) -> PredictionErrorReport:
        return neural_prediction_error(self.tensor, actual_state, level=level)

    def snn_firing_activation(self, tau: float = 1.0, gamma: float = 0.5,
                              theta_base: float = 0.0) -> FiringActivation:
        return snn_firing_activation(self.tensor, tau=tau, gamma=gamma,
                                     theta_base=theta_base)

    @staticmethod
    def ground_truth_pytest(test_dir: str = 'tests',
                            baseline_pass_rate: float = 1.0) -> TestJumpEvent:
        return ground_truth_pytest(test_dir=test_dir,
                                   baseline_pass_rate=baseline_pass_rate)

    def check_feed_health(self, feed_status: Optional[dict] = None,
                          max_staleness: float = 60.0) -> FeedHealth:
        return check_feed_health(self.tensor, feed_status=feed_status,
                                 max_staleness=max_staleness)
