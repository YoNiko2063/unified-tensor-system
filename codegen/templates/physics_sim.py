"""Physics simulation templates — pure Rust RK4 integrators and Koopman EDMD.

Follows rust-physics-kernel/src/lib.rs pattern:
  Layer 1: Pure core (stack-allocated state, no heap in inner loop)
  Layer 2: Vec<f64> wrapper for Python interop

All templates have E_borrow < D_SEP=0.43.
No external crates (ndarray, nalgebra, etc.) — flat Vec<f64> matrix ops.
"""

from __future__ import annotations

from typing import Dict

from codegen.intent_spec import BorrowProfile
from codegen.template_registry import RustTemplate, TemplateRegistry


def rk4_integrator(params: Dict) -> str:
    """Generic 4th-order Runge-Kutta integrator.

    Pure functional on owned Vec<f64> state.
    BV = (0.10, 0.00, 0.00, 0.00, 0.00, 0.00), E ≈ 0.025
    """
    state_dim = params.get("state_dim", 2)
    dt = params.get("dt", 0.01)
    return f'''\
/// Generic RK4 integrator for state_dim={state_dim} ODE systems.
///
/// The caller provides `rhs: fn(&[f64], f64) -> Vec<f64>` as a function pointer.
/// Pure functional: takes owned Vec<f64> state, returns new owned Vec<f64>.
/// No external crates — flat Vec<f64> arithmetic.

pub const STATE_DIM: usize = {state_dim};
pub const DEFAULT_DT: f64 = {dt};

/// Add two equal-length slices element-wise, scaled: result = a + scale * b.
fn add_scaled(a: &[f64], b: &[f64], scale: f64) -> Vec<f64> {{
    a.iter().zip(b.iter()).map(|(ai, bi)| ai + scale * bi).collect()
}}

/// Single RK4 step — pure functional.
///
/// # Arguments
/// * `state` — current state vector (length = STATE_DIM)
/// * `t`     — current time
/// * `dt`    — time step
/// * `rhs`   — right-hand-side function f(state, t) -> derivative
///
/// # Returns
/// New state after one RK4 step.
pub fn rk4_step(
    state: Vec<f64>,
    t: f64,
    dt: f64,
    rhs: fn(&[f64], f64) -> Vec<f64>,
) -> Vec<f64> {{
    let k1 = rhs(&state, t);
    let s2 = add_scaled(&state, &k1, 0.5 * dt);
    let k2 = rhs(&s2, t + 0.5 * dt);
    let s3 = add_scaled(&state, &k2, 0.5 * dt);
    let k3 = rhs(&s3, t + 0.5 * dt);
    let s4 = add_scaled(&state, &k3, dt);
    let k4 = rhs(&s4, t + dt);
    state.iter()
        .enumerate()
        .map(|(i, &si)| si + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}}

/// Integrate over n_steps, returning full trajectory as Vec<Vec<f64>>.
pub fn integrate(
    initial: Vec<f64>,
    t0: f64,
    dt: f64,
    n_steps: usize,
    rhs: fn(&[f64], f64) -> Vec<f64>,
) -> Vec<Vec<f64>> {{
    let mut traj = Vec::with_capacity(n_steps + 1);
    let mut state = initial;
    let mut t = t0;
    traj.push(state.clone());
    for _ in 0..n_steps {{
        state = rk4_step(state, t, dt, rhs);
        t += dt;
        traj.push(state.clone());
    }}
    traj
}}

#[cfg(test)]
mod tests {{
    use super::*;

    fn exponential_decay(state: &[f64], _t: f64) -> Vec<f64> {{
        // dx/dt = -x  →  x(t) = x0 * exp(-t)
        vec![-state[0]]
    }}

    #[test]
    fn test_rk4_exponential_decay() {{
        let state = vec![1.0];
        let result = rk4_step(state, 0.0, 0.01, exponential_decay);
        // x(0.01) = exp(-0.01) ≈ 0.99005
        assert!((result[0] - (-0.01_f64).exp()).abs() < 1e-7);
    }}

    #[test]
    fn test_integrate_length() {{
        let traj = integrate(vec![1.0, 0.0], 0.0, {dt}, 100, |s, _t| vec![-s[0], -s[1]]);
        assert_eq!(traj.len(), 101);
        assert_eq!(traj[0].len(), {state_dim});
    }}

    #[test]
    fn test_rk4_state_dim() {{
        // Test with {state_dim}-dimensional zero state — should stay at zero
        let zero_state = vec![0.0; {state_dim}];
        let result = rk4_step(zero_state, 0.0, {dt}, |s, _t| s.to_vec());
        assert_eq!(result.len(), {state_dim});
        for v in &result {{
            assert!(v.abs() < 1e-15);
        }}
    }}
}}
'''


def harmonic_oscillator(params: Dict) -> str:
    """Second-order harmonic oscillator: ẍ + 2ζω₀ẋ + ω₀²x = 0.

    Pure functional RK4. State = [x, v].
    BV = (0.10, 0.00, 0.00, 0.00, 0.00, 0.00), E ≈ 0.025
    """
    omega0 = params.get("omega0", 1.0)
    zeta = params.get("zeta", 0.1)
    dt = params.get("dt", 0.01)
    return f'''\
/// Harmonic oscillator: x'' + 2*zeta*omega0*x' + omega0^2*x = 0.
///
/// State vector: [x, v] where v = dx/dt.
/// Pure functional RK4 — owned Vec<f64>, no lifetime annotations.

pub const OMEGA0: f64 = {omega0};
pub const ZETA: f64 = {zeta};
pub const DT: f64 = {dt};

/// RHS of harmonic oscillator ODE.
///
/// Returns [dx/dt, dv/dt] = [v, -2ζω₀v - ω₀²x].
#[inline(always)]
pub fn harmonic_rhs(state: &[f64], omega0: f64, zeta: f64) -> Vec<f64> {{
    let x = state[0];
    let v = state[1];
    vec![
        v,
        -2.0 * zeta * omega0 * v - omega0 * omega0 * x,
    ]
}}

/// Single RK4 step for harmonic oscillator.
pub fn rk4_step(state: Vec<f64>, omega0: f64, zeta: f64, dt: f64) -> Vec<f64> {{
    let k1 = harmonic_rhs(&state, omega0, zeta);
    let s2: Vec<f64> = state.iter().zip(k1.iter()).map(|(s, k)| s + 0.5 * dt * k).collect();
    let k2 = harmonic_rhs(&s2, omega0, zeta);
    let s3: Vec<f64> = state.iter().zip(k2.iter()).map(|(s, k)| s + 0.5 * dt * k).collect();
    let k3 = harmonic_rhs(&s3, omega0, zeta);
    let s4: Vec<f64> = state.iter().zip(k3.iter()).map(|(s, k)| s + dt * k).collect();
    let k4 = harmonic_rhs(&s4, omega0, zeta);
    state.iter()
        .enumerate()
        .map(|(i, &si)| si + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}}

/// Simulate harmonic oscillator for n_steps.
pub fn simulate(
    x0: f64,
    v0: f64,
    omega0: f64,
    zeta: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<[f64; 2]> {{
    let mut traj: Vec<[f64; 2]> = Vec::with_capacity(n_steps + 1);
    let mut state = vec![x0, v0];
    traj.push([state[0], state[1]]);
    for _ in 0..n_steps {{
        state = rk4_step(state, omega0, zeta, dt);
        traj.push([state[0], state[1]]);
    }}
    traj
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_undamped_energy_conservation() {{
        // Undamped: zeta=0. Energy = 0.5*v^2 + 0.5*omega0^2*x^2 should be conserved.
        let traj = simulate(1.0, 0.0, {omega0}, 0.0, {dt}, 1000);
        let e0 = 0.5 * {omega0} * {omega0} * traj[0][0] * traj[0][0]
                 + 0.5 * traj[0][1] * traj[0][1];
        let ef = 0.5 * {omega0} * {omega0} * traj[1000][0] * traj[1000][0]
                 + 0.5 * traj[1000][1] * traj[1000][1];
        assert!((e0 - ef).abs() / (e0 + 1e-15) < 1e-6,
            "Energy drift: {{e0}} -> {{ef}}");
    }}

    #[test]
    fn test_overdamped_decay() {{
        // zeta > 1: no oscillation, monotone decay
        let traj = simulate(1.0, 0.0, {omega0}, 2.0, {dt}, 5000);
        let final_x = traj.last().unwrap()[0].abs();
        assert!(final_x < 0.01, "Overdamped should decay: x={{final_x}}");
    }}

    #[test]
    fn test_default_params() {{
        let traj = simulate(1.0, 0.0, OMEGA0, ZETA, DT, 100);
        assert_eq!(traj.len(), 101);
    }}
}}
'''


def duffing_sim(params: Dict) -> str:
    """Duffing oscillator: ẍ + δẋ + αx + βx³ = F·cos(Ωt).

    Pure functional RK4, shared reference to parameter struct.
    BV = (0.15, 0.10, 0.00, 0.00, 0.00, 0.00), E ≈ 0.056
    """
    alpha = params.get("alpha", 1.0)
    beta = params.get("beta", 0.1)
    delta = params.get("delta", 0.1)
    F = params.get("F", 0.0)
    omega = params.get("omega", 1.0)
    dt = params.get("dt", 0.01)
    return f'''\
/// Duffing oscillator simulation.
///
/// Equation: x\\'\\' + delta*x\\' + alpha*x + beta*x^3 = F*cos(omega*t)
/// State: [x, v] where v = dx/dt.
/// Pure functional RK4. Parameters passed as shared reference to DuffingParams struct.

/// Duffing oscillator parameters.
#[derive(Clone, Copy, Debug)]
pub struct DuffingParams {{
    pub alpha: f64,   // linear stiffness
    pub beta: f64,    // cubic nonlinearity (hardening > 0, softening < 0)
    pub delta: f64,   // damping coefficient
    pub f_amp: f64,   // forcing amplitude F
    pub omega: f64,   // forcing frequency Omega
}}

impl DuffingParams {{
    /// Default parameters: alpha={alpha}, beta={beta}, delta={delta}, F={F}, omega={omega}.
    pub fn default_params() -> Self {{
        DuffingParams {{
            alpha: {alpha},
            beta: {beta},
            delta: {delta},
            f_amp: {F},
            omega: {omega},
        }}
    }}
}}

/// RHS: [dx/dt, dv/dt] for Duffing equation.
#[inline(always)]
pub fn duffing_rhs(state: &[f64], t: f64, p: &DuffingParams) -> [f64; 2] {{
    let x = state[0];
    let v = state[1];
    [
        v,
        -p.delta * v - p.alpha * x - p.beta * x * x * x + p.f_amp * (p.omega * t).cos(),
    ]
}}

/// Single RK4 step for Duffing oscillator.
pub fn rk4_step(state: &[f64; 2], t: f64, dt: f64, p: &DuffingParams) -> [f64; 2] {{
    let k1 = duffing_rhs(state, t, p);
    let s2 = [state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1]];
    let k2 = duffing_rhs(&s2, t + 0.5 * dt, p);
    let s3 = [state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1]];
    let k3 = duffing_rhs(&s3, t + 0.5 * dt, p);
    let s4 = [state[0] + dt * k3[0], state[1] + dt * k3[1]];
    let k4 = duffing_rhs(&s4, t + dt, p);
    [
        state[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        state[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
    ]
}}

/// Simulate Duffing oscillator for n_steps. Returns Vec of [x, v] states.
pub fn simulate(
    x0: f64,
    v0: f64,
    dt: f64,
    n_steps: usize,
    p: &DuffingParams,
) -> Vec<[f64; 2]> {{
    let mut traj = Vec::with_capacity(n_steps + 1);
    let mut state = [x0, v0];
    traj.push(state);
    for i in 0..n_steps {{
        let t = i as f64 * dt;
        state = rk4_step(&state, t, dt, p);
        traj.push(state);
    }}
    traj
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_unforced_decay() {{
        // Unforced (F=0), damped: should decay to zero.
        let p = DuffingParams {{ alpha: {alpha}, beta: {beta}, delta: 0.5, f_amp: 0.0, omega: {omega} }};
        let traj = simulate(1.0, 0.0, {dt}, 10000, &p);
        let final_x = traj.last().unwrap()[0].abs();
        assert!(final_x < 0.05, "Should decay: x={{final_x}}");
    }}

    #[test]
    fn test_energy_undamped_unforced() {{
        // Undamped, unforced, no cubic: pure harmonic → energy conserved.
        let p = DuffingParams {{ alpha: {alpha}, beta: 0.0, delta: 0.0, f_amp: 0.0, omega: 0.0 }};
        let traj = simulate(1.0, 0.0, 0.001, 1000, &p);
        let e = |s: &[f64; 2]| 0.5 * s[1] * s[1] + 0.5 * {alpha} * s[0] * s[0];
        let e0 = e(&traj[0]);
        let ef = e(traj.last().unwrap());
        assert!((e0 - ef).abs() / (e0 + 1e-15) < 1e-6,
            "Energy drift: {{e0}} -> {{ef}}");
    }}

    #[test]
    fn test_default_params_accessible() {{
        let p = DuffingParams::default_params();
        assert_eq!(p.alpha, {alpha});
        assert_eq!(p.beta, {beta});
        assert_eq!(p.delta, {delta});
    }}
}}
'''


def rlc_circuit_sim(params: Dict) -> str:
    """RLC circuit simulation: L·dI/dt = V - R·I - q/C, dq/dt = I.

    Shared reference to circuit parameters.
    BV = (0.15, 0.10, 0.00, 0.00, 0.00, 0.00), E ≈ 0.056
    """
    R = params.get("R", 1.0)
    L = params.get("L", 1.0)
    C = params.get("C", 1.0)
    dt = params.get("dt", 0.001)
    return f'''\
/// RLC circuit simulation.
///
/// KVL: V_source = V_R + V_L + V_C
///   L * dI/dt = V_source - R*I - q/C
///   dq/dt = I
///
/// State: [q (charge), I (current)]
/// Parameters passed as shared reference to RLCParams struct.

/// RLC circuit parameters.
#[derive(Clone, Copy, Debug)]
pub struct RLCParams {{
    pub r: f64,         // resistance (Ohm)
    pub l: f64,         // inductance (Henry)
    pub c: f64,         // capacitance (Farad)
    pub v_source: f64,  // source voltage (V), 0 for free response
}}

impl RLCParams {{
    /// Default: R={R}, L={L}, C={C}, V_source=0.
    pub fn default_params() -> Self {{
        RLCParams {{ r: {R}, l: {L}, c: {C}, v_source: 0.0 }}
    }}

    /// Natural frequency omega0 = 1/sqrt(L*C).
    pub fn omega0(&self) -> f64 {{
        1.0 / (self.l * self.c).sqrt()
    }}

    /// Quality factor Q = omega0*L/R.
    pub fn quality_factor(&self) -> f64 {{
        self.omega0() * self.l / self.r
    }}
}}

/// RHS: [dq/dt, dI/dt].
///
/// state[0] = q (charge), state[1] = I (current).
#[inline(always)]
pub fn rlc_rhs(state: &[f64], p: &RLCParams) -> [f64; 2] {{
    let q = state[0];
    let i = state[1];
    [
        i,                                            // dq/dt = I
        (p.v_source - p.r * i - q / p.c) / p.l,    // dI/dt = (V - R*I - q/C) / L
    ]
}}

/// Single RK4 step for RLC circuit.
pub fn rk4_step(state: &[f64; 2], dt: f64, p: &RLCParams) -> [f64; 2] {{
    let k1 = rlc_rhs(state, p);
    let s2 = [state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1]];
    let k2 = rlc_rhs(&s2, p);
    let s3 = [state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1]];
    let k3 = rlc_rhs(&s3, p);
    let s4 = [state[0] + dt * k3[0], state[1] + dt * k3[1]];
    let k4 = rlc_rhs(&s4, p);
    [
        state[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        state[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
    ]
}}

/// Simulate RLC circuit for n_steps. Initial: [q0, I0].
pub fn simulate(
    q0: f64,
    i0: f64,
    dt: f64,
    n_steps: usize,
    p: &RLCParams,
) -> Vec<[f64; 2]> {{
    let mut traj = Vec::with_capacity(n_steps + 1);
    let mut state = [q0, i0];
    traj.push(state);
    for _ in 0..n_steps {{
        state = rk4_step(&state, dt, p);
        traj.push(state);
    }}
    traj
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_free_response_decay() {{
        // High damping (overdamped): charge should decay to zero.
        let p = RLCParams {{ r: 10.0, l: {L}, c: {C}, v_source: 0.0 }};
        let traj = simulate(1.0, 0.0, {dt}, 10000, &p);
        let final_q = traj.last().unwrap()[0].abs();
        assert!(final_q < 0.01, "Should decay: q={{final_q}}");
    }}

    #[test]
    fn test_energy_lossless_approx() {{
        // R=0 (lossless LC): electromagnetic energy should be conserved.
        let p = RLCParams {{ r: 0.0, l: {L}, c: {C}, v_source: 0.0 }};
        let traj = simulate(1.0, 0.0, 0.0001, 1000, &p);
        // Energy = q^2/(2C) + L*I^2/2
        let e = |s: &[f64; 2]| s[0] * s[0] / (2.0 * {C}) + {L} * s[1] * s[1] / 2.0;
        let e0 = e(&traj[0]);
        let ef = e(traj.last().unwrap());
        assert!((e0 - ef).abs() / (e0 + 1e-15) < 1e-4,
            "Energy drift: {{e0}} -> {{ef}}");
    }}

    #[test]
    fn test_omega0_quality_factor() {{
        let p = RLCParams::default_params();
        let omega0 = p.omega0();
        assert!(omega0 > 0.0);
        let q = p.quality_factor();
        assert!(q > 0.0);
    }}
}}
'''


def koopman_edmd_kernel(params: Dict) -> str:
    """EDMD observable lifting kernel: K ≈ (Ψ'Ψ)⁻¹Ψ'Ψ_next.

    Mutable output matrix, shared reference to snapshot data.
    BV = (0.20, 0.10, 0.00, 0.20, 0.00, 0.00), E ≈ 0.108
    """
    state_dim = params.get("state_dim", 2)
    n_obs = params.get("n_observables", 6)
    return f'''\
/// Koopman EDMD (Extended Dynamic Mode Decomposition) kernel.
///
/// Computes Koopman matrix K ≈ (Psi\\'*Psi)^{{-1}} * Psi\\'*Psi_next
/// using observable lifting with monomials up to degree 2.
///
/// State dimension: {state_dim}, observables: {n_obs}.
/// All matrices are flat Vec<f64> in row-major order.
/// No external crates — pure Rust matrix arithmetic.

pub const STATE_DIM: usize = {state_dim};
pub const N_OBSERVABLES: usize = {n_obs};

/// Lift a state vector to observable space using monomial basis.
///
/// Observable basis: [1, x1, x2, ..., x_d, x1^2, x1*x2, ..., x_d^2]
/// Truncated to N_OBSERVABLES entries.
pub fn lift_state(state: &[f64]) -> Vec<f64> {{
    let mut obs = Vec::with_capacity(N_OBSERVABLES);
    // Bias term
    obs.push(1.0);
    // Linear terms
    for &xi in state.iter().take(STATE_DIM) {{
        if obs.len() >= N_OBSERVABLES {{ break; }}
        obs.push(xi);
    }}
    // Quadratic terms
    'outer: for i in 0..STATE_DIM {{
        for j in i..STATE_DIM {{
            if obs.len() >= N_OBSERVABLES {{ break 'outer; }}
            obs.push(state[i] * state[j]);
        }}
    }}
    // Pad with zeros if needed
    while obs.len() < N_OBSERVABLES {{
        obs.push(0.0);
    }}
    obs
}}

/// Matrix multiply: C = A * B where A is (m x k), B is (k x n).
/// Matrices are flat Vec<f64> in row-major order.
pub fn mat_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {{
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {{
        for l in 0..k {{
            for j in 0..n {{
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }}
        }}
    }}
    c
}}

/// Matrix transpose: B = A^T where A is (m x n).
pub fn mat_transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {{
    let mut b = vec![0.0_f64; n * m];
    for i in 0..m {{
        for j in 0..n {{
            b[j * m + i] = a[i * n + j];
        }}
    }}
    b
}}

/// Invert a square matrix using Gauss-Jordan elimination.
/// Returns None if matrix is singular (determinant < 1e-14).
pub fn mat_inv(a: &[f64], n: usize) -> Option<Vec<f64>> {{
    let mut aug = vec![0.0_f64; n * 2 * n];
    // Fill [A | I]
    for i in 0..n {{
        for j in 0..n {{
            aug[i * 2 * n + j] = a[i * n + j];
        }}
        aug[i * 2 * n + n + i] = 1.0;
    }}
    for col in 0..n {{
        // Pivot selection
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {{
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {{
                max_val = v;
                max_row = row;
            }}
        }}
        if max_val < 1e-14 {{
            return None; // singular
        }}
        // Swap rows
        if max_row != col {{
            for j in 0..2 * n {{
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }}
        }}
        // Scale pivot row
        let pivot = aug[col * 2 * n + col];
        for j in 0..2 * n {{
            aug[col * 2 * n + j] /= pivot;
        }}
        // Eliminate column
        for row in 0..n {{
            if row == col {{ continue; }}
            let factor = aug[row * 2 * n + col];
            for j in 0..2 * n {{
                let delta = factor * aug[col * 2 * n + j];
                aug[row * 2 * n + j] -= delta;
            }}
        }}
    }}
    // Extract inverse (right half of augmented matrix)
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {{
        for j in 0..n {{
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }}
    }}
    Some(inv)
}}

/// Compute EDMD Koopman matrix K from snapshot pairs.
///
/// # Arguments
/// * `snapshots`      — current states, shape (n_snapshots, STATE_DIM), row-major
/// * `snapshots_next` — next states,    shape (n_snapshots, STATE_DIM), row-major
/// * `n_snapshots`    — number of snapshot pairs
///
/// # Returns
/// Koopman matrix K of shape (N_OBSERVABLES, N_OBSERVABLES), row-major.
/// Returns None if Psi\\'*Psi is singular.
pub fn compute_edmd(
    snapshots: &[f64],
    snapshots_next: &[f64],
    n_snapshots: usize,
) -> Option<Vec<f64>> {{
    let p = N_OBSERVABLES;

    // Build Psi (n_snapshots x p) and Psi_next (n_snapshots x p)
    let mut psi      = vec![0.0_f64; n_snapshots * p];
    let mut psi_next = vec![0.0_f64; n_snapshots * p];

    for s in 0..n_snapshots {{
        let state      = &snapshots[s * STATE_DIM..(s + 1) * STATE_DIM];
        let state_next = &snapshots_next[s * STATE_DIM..(s + 1) * STATE_DIM];
        let obs      = lift_state(state);
        let obs_next = lift_state(state_next);
        for k in 0..p {{
            psi[s * p + k]      = obs[k];
            psi_next[s * p + k] = obs_next[k];
        }}
    }}

    // Psi_T = Psi^T  (p x n_snapshots)
    let psi_t = mat_transpose(&psi, n_snapshots, p);

    // G = Psi^T * Psi  (p x p)
    let g = mat_mul(&psi_t, &psi, p, n_snapshots, p);

    // A = Psi^T * Psi_next  (p x p)
    let a = mat_mul(&psi_t, &psi_next, p, n_snapshots, p);

    // K = G^{{-1}} * A
    let g_inv = mat_inv(&g, p)?;
    Some(mat_mul(&g_inv, &a, p, p, p))
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_lift_state_dimension() {{
        let state = vec![1.0_f64; STATE_DIM];
        let obs = lift_state(&state);
        assert_eq!(obs.len(), N_OBSERVABLES);
    }}

    #[test]
    fn test_lift_bias_term() {{
        let state = vec![2.0_f64; STATE_DIM];
        let obs = lift_state(&state);
        // First element should always be 1 (bias)
        assert_eq!(obs[0], 1.0);
    }}

    #[test]
    fn test_mat_mul_identity() {{
        // I * I = I (3x3)
        let eye: Vec<f64> = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let result = mat_mul(&eye, &eye, 3, 3, 3);
        for i in 0..3 {{
            for j in 0..3 {{
                let expected = if i == j {{ 1.0 }} else {{ 0.0 }};
                assert!((result[i * 3 + j] - expected).abs() < 1e-12);
            }}
        }}
    }}

    #[test]
    fn test_mat_inv_2x2() {{
        // [[2, 0], [0, 4]] → inv = [[0.5, 0], [0, 0.25]]
        let a = vec![2.0, 0.0, 0.0, 4.0];
        let inv = mat_inv(&a, 2).expect("should be invertible");
        assert!((inv[0] - 0.5).abs() < 1e-12);
        assert!((inv[3] - 0.25).abs() < 1e-12);
    }}

    #[test]
    fn test_edmd_linear_system() {{
        // Linear system x_{{n+1}} = 0.9 * x_n → Koopman should recover scaling.
        let n = 50;
        let mut snaps      = vec![0.0_f64; n * STATE_DIM];
        let mut snaps_next = vec![0.0_f64; n * STATE_DIM];
        for i in 0..n {{
            let x = (i as f64 + 1.0) * 0.1;
            snaps[i * STATE_DIM] = x;
            snaps_next[i * STATE_DIM] = 0.9 * x;
            if STATE_DIM > 1 {{
                snaps[i * STATE_DIM + 1] = 0.0;
                snaps_next[i * STATE_DIM + 1] = 0.0;
            }}
        }}
        let k_opt = compute_edmd(&snaps, &snaps_next, n);
        assert!(k_opt.is_some(), "EDMD should succeed for linear system");
        let k = k_opt.unwrap();
        assert_eq!(k.len(), N_OBSERVABLES * N_OBSERVABLES);
    }}
}}
'''


# ── Template objects ─────────────────────────────────────────────────────────

RK4_INTEGRATOR_TEMPLATE = RustTemplate(
    name="rk4_integrator",
    domain="physics",
    operation="rk4_integrator",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=rk4_integrator,
    description="Generic RK4 integrator — owned Vec<f64> state, function pointer RHS",
)

HARMONIC_OSCILLATOR_TEMPLATE = RustTemplate(
    name="harmonic_oscillator",
    domain="physics",
    operation="harmonic_oscillator",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=harmonic_oscillator,
    description="Second-order harmonic oscillator RK4 — pure functional",
)

DUFFING_SIM_TEMPLATE = RustTemplate(
    name="duffing_sim",
    domain="physics",
    operation="duffing_sim",
    borrow_profile=BorrowProfile.SHARED_REFERENCE,
    design_bv=(0.15, 0.10, 0.00, 0.00, 0.00, 0.00),
    render=duffing_sim,
    description="Duffing oscillator RK4 — shared DuffingParams reference",
)

RLC_CIRCUIT_SIM_TEMPLATE = RustTemplate(
    name="rlc_circuit_sim",
    domain="physics",
    operation="rlc_circuit_sim",
    borrow_profile=BorrowProfile.SHARED_REFERENCE,
    design_bv=(0.15, 0.10, 0.00, 0.00, 0.00, 0.00),
    render=rlc_circuit_sim,
    description="RLC circuit KVL/KCL RK4 — shared RLCParams reference",
)

KOOPMAN_EDMD_TEMPLATE = RustTemplate(
    name="koopman_edmd_kernel",
    domain="physics",
    operation="koopman_edmd_kernel",
    borrow_profile=BorrowProfile.MUTABLE_OUTPUT,
    design_bv=(0.20, 0.10, 0.00, 0.20, 0.00, 0.00),
    render=koopman_edmd_kernel,
    description="EDMD Koopman matrix solve — mutable output, shared snapshot data",
)

ALL_PHYSICS_TEMPLATES = [
    RK4_INTEGRATOR_TEMPLATE,
    HARMONIC_OSCILLATOR_TEMPLATE,
    DUFFING_SIM_TEMPLATE,
    RLC_CIRCUIT_SIM_TEMPLATE,
    KOOPMAN_EDMD_TEMPLATE,
]


def register_all(registry: TemplateRegistry) -> None:
    """Register all physics simulation templates into a registry."""
    for t in ALL_PHYSICS_TEMPLATES:
        registry.register(t)
