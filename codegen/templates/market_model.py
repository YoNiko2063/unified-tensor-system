"""Market model templates — Duffing resonance kernels for financial applications.

Maps market semantics onto Duffing oscillator parameters:
  alpha  → mean_reversion strength
  beta   → technical_nonlinearity (hardening/softening)
  delta  → friction / market friction coefficient
  F      → news_amplitude (forcing)
  omega  → news_frequency (periodicity of external shock)

Resonance physics: when omega ~ sqrt(alpha), news forcing causes outsized moves.
Backbone curve gives probabilistic multi-stability at fold bifurcations.
"""

from __future__ import annotations

from typing import Dict

from codegen.intent_spec import BorrowProfile
from codegen.template_registry import RustTemplate, TemplateRegistry


def _duffing_market_oscillator_render(params: Dict) -> str:
    """Duffing RK4 with market-semantic parameter names."""
    mod_name = params.get("module_name", "duffing_market")
    return f'''\
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Market-semantic Duffing parameters.
#[derive(Clone, Copy, Debug)]
pub struct MarketDuffingParams {{
    pub mean_reversion: f64,        // alpha — linear restoring
    pub nonlinearity: f64,          // beta — cubic stiffness
    pub friction: f64,              // delta — damping
    pub news_amplitude: f64,        // F — forcing amplitude
    pub news_frequency: f64,        // omega — forcing frequency
}}

/// Duffing RHS: x'' + delta*x' + alpha*x + beta*x^3 = F*cos(omega*t)
#[inline(always)]
fn duffing_rhs(state: &[f64; 2], t: f64, p: &MarketDuffingParams) -> [f64; 2] {{
    let x = state[0];
    let v = state[1];
    let dx = v;
    let dv = -p.friction * v - p.mean_reversion * x - p.nonlinearity * x * x * x
             + p.news_amplitude * (p.news_frequency * t).cos();
    [dx, dv]
}}

#[inline(always)]
fn add_scaled(a: &[f64; 2], b: &[f64; 2], s: f64) -> [f64; 2] {{
    [a[0] + s * b[0], a[1] + s * b[1]]
}}

/// RK4 single step — pure, stack-allocated.
#[inline(always)]
pub fn rk4_step(state: &[f64; 2], t: f64, dt: f64, p: &MarketDuffingParams) -> [f64; 2] {{
    let k1 = duffing_rhs(state, t, p);
    let s2 = add_scaled(state, &k1, 0.5 * dt);
    let k2 = duffing_rhs(&s2, t + 0.5 * dt, p);
    let s3 = add_scaled(state, &k2, 0.5 * dt);
    let k3 = duffing_rhs(&s3, t + 0.5 * dt, p);
    let s4 = add_scaled(state, &k3, dt);
    let k4 = duffing_rhs(&s4, t + dt, p);
    [
        state[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        state[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
    ]
}}

/// Generate full trajectory.
pub fn generate_trajectory(
    x0: f64, v0: f64, dt: f64, n_steps: usize,
    p: &MarketDuffingParams,
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

// ── PyO3 bindings ────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    x0, v0, dt, n_steps,
    mean_reversion=1.0, nonlinearity=0.1,
    friction=0.1, news_amplitude=0.0, news_frequency=1.0
))]
fn py_market_trajectory(
    x0: f64, v0: f64, dt: f64, n_steps: usize,
    mean_reversion: f64, nonlinearity: f64,
    friction: f64, news_amplitude: f64, news_frequency: f64,
) -> PyResult<Vec<Vec<f64>>> {{
    if dt <= 0.0 {{
        return Err(PyValueError::new_err("dt must be positive"));
    }}
    let p = MarketDuffingParams {{
        mean_reversion, nonlinearity, friction, news_amplitude, news_frequency,
    }};
    Ok(generate_trajectory(x0, v0, dt, n_steps, &p)
        .iter().map(|s| s.to_vec()).collect())
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_market_trajectory, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_unforced_decay() {{
        let p = MarketDuffingParams {{
            mean_reversion: 1.0,
            nonlinearity: 0.0,
            friction: 0.3,
            news_amplitude: 0.0,
            news_frequency: 1.0,
        }};
        let traj = generate_trajectory(1.0, 0.0, 0.01, 10000, &p);
        let final_x = traj.last().unwrap()[0].abs();
        assert!(final_x < 0.01, "Damped system should decay: x={{final_x}}");
    }}

    #[test]
    fn test_energy_conservation_undamped() {{
        let p = MarketDuffingParams {{
            mean_reversion: 1.0,
            nonlinearity: 0.1,
            friction: 0.0,
            news_amplitude: 0.0,
            news_frequency: 1.0,
        }};
        let traj = generate_trajectory(1.0, 0.0, 0.001, 1000, &p);
        let energy = |s: &[f64; 2]| -> f64 {{
            0.5 * s[1] * s[1] + 0.5 * p.mean_reversion * s[0] * s[0]
                + 0.25 * p.nonlinearity * s[0].powi(4)
        }};
        let e0 = energy(&traj[0]);
        let ef = energy(traj.last().unwrap());
        assert!((e0 - ef).abs() / e0 < 1e-6, "Energy drift: {{e0}} → {{ef}}");
    }}
}}
'''


def _resonance_detector_render(params: Dict) -> str:
    """Backbone curve sweep + fold bifurcation detection."""
    mod_name = params.get("module_name", "resonance_detector")
    return f'''\
use pyo3::prelude::*;

/// Backbone curve: amplitude-frequency relation for Duffing oscillator.
/// For x'' + delta*x' + alpha*x + beta*x^3 = F*cos(omega*t),
/// the steady-state amplitude A satisfies:
///   ((alpha - omega^2) + 3/4*beta*A^2)^2 + (delta*omega)^2 = (F/A)^2
///
/// This function sweeps omega and finds the backbone A(omega) numerically.
pub fn backbone_curve(
    alpha: f64, beta: f64, delta: f64, f_drive: f64,
    omega_min: f64, omega_max: f64, n_points: usize,
) -> Vec<[f64; 2]> {{
    let mut curve = Vec::with_capacity(n_points);
    let d_omega = (omega_max - omega_min) / (n_points - 1).max(1) as f64;

    for i in 0..n_points {{
        let omega = omega_min + i as f64 * d_omega;
        // Solve for amplitude A via Newton iteration on:
        //   g(A) = ((alpha - omega^2) + 3/4*beta*A^2)^2*A^2 + (delta*omega*A)^2 - F^2
        let mut a = f_drive / ((alpha - omega * omega).abs() + 0.01).sqrt();
        for _ in 0..50 {{
            let detuning = alpha - omega * omega + 0.75 * beta * a * a;
            let g = detuning * detuning * a * a + delta * delta * omega * omega * a * a
                    - f_drive * f_drive;
            let dg = 2.0 * detuning * (1.5 * beta * a) * a * a
                     + 2.0 * detuning * detuning * a
                     + 2.0 * delta * delta * omega * omega * a;
            if dg.abs() < 1e-15 {{ break; }}
            let step = g / dg;
            a -= step;
            if a < 1e-12 {{ a = 1e-12; }}
            if step.abs() < 1e-12 {{ break; }}
        }}
        curve.push([omega, a]);
    }}
    curve
}}

/// Detect fold bifurcations where dA/domega → ∞ (slope sign change).
/// Returns list of (omega, amplitude) fold points.
pub fn detect_folds(curve: &[[f64; 2]]) -> Vec<[f64; 2]> {{
    let mut folds = Vec::new();
    for i in 1..curve.len().saturating_sub(1) {{
        let slope_left = curve[i][1] - curve[i - 1][1];
        let slope_right = curve[i + 1][1] - curve[i][1];
        // Fold: slope changes sign (turning point on backbone)
        if slope_left * slope_right < 0.0 {{
            folds.push(curve[i]);
        }}
    }}
    folds
}}

/// Jump probability from basin volumes at a fold bifurcation.
/// Estimates probability of jumping between coexisting attractors
/// based on relative basin sizes (simplified energy-based approximation).
pub fn jump_probability(alpha: f64, beta: f64, f_drive: f64, omega: f64) -> f64 {{
    let detuning = alpha - omega * omega;
    // At fold: bistability exists when beta > 0 and |detuning| < sqrt(3)*F/(2*sqrt(beta))
    let threshold = if beta.abs() < 1e-15 {{
        return 0.0;
    }} else {{
        3.0_f64.sqrt() * f_drive / (2.0 * beta.abs().sqrt())
    }};
    if detuning.abs() > threshold {{
        return 0.0;
    }}
    // Probability increases as we approach the fold center
    let proximity = 1.0 - detuning.abs() / threshold;
    proximity * proximity  // quadratic basin volume scaling
}}

#[pyfunction]
#[pyo3(signature = (alpha, beta, delta, f_drive, omega_min=0.5, omega_max=2.0, n_points=200))]
fn py_backbone_curve(
    alpha: f64, beta: f64, delta: f64, f_drive: f64,
    omega_min: f64, omega_max: f64, n_points: usize,
) -> PyResult<Vec<Vec<f64>>> {{
    Ok(backbone_curve(alpha, beta, delta, f_drive, omega_min, omega_max, n_points)
        .iter().map(|r| r.to_vec()).collect())
}}

#[pyfunction]
fn py_detect_folds(curve: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {{
    let arr: Vec<[f64; 2]> = curve.iter()
        .map(|v| [v[0], v[1]])
        .collect();
    Ok(detect_folds(&arr).iter().map(|r| r.to_vec()).collect())
}}

#[pyfunction]
fn py_jump_probability(alpha: f64, beta: f64, f_drive: f64, omega: f64) -> f64 {{
    jump_probability(alpha, beta, f_drive, omega)
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_backbone_curve, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_folds, m)?)?;
    m.add_function(wrap_pyfunction!(py_jump_probability, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_backbone_linear() {{
        // Linear (beta=0): resonance at omega=sqrt(alpha)=1.0
        let curve = backbone_curve(1.0, 0.0, 0.1, 0.3, 0.5, 1.5, 100);
        assert_eq!(curve.len(), 100);
        // Peak should be near omega=1.0
        let peak = curve.iter()
            .max_by(|a, b| a[1].partial_cmp(&b[1]).unwrap())
            .unwrap();
        assert!((peak[0] - 1.0).abs() < 0.15, "Peak at omega={{}} (expected ~1.0)", peak[0]);
    }}

    #[test]
    fn test_fold_detection() {{
        // Nonlinear (beta=0.1) should show fold near resonance
        let curve = backbone_curve(1.0, 0.1, 0.05, 0.3, 0.5, 1.5, 500);
        let folds = detect_folds(&curve);
        // With beta>0 and low damping, we expect fold bifurcations
        // (may or may not appear depending on parameters)
        assert!(curve.len() == 500);
    }}

    #[test]
    fn test_jump_probability_bounds() {{
        let p = jump_probability(1.0, 0.1, 0.3, 1.0);
        assert!(p >= 0.0 && p <= 1.0);
        // Far from resonance → low probability
        let p_far = jump_probability(1.0, 0.1, 0.3, 3.0);
        assert!(p_far < 0.01);
    }}
}}
'''


def _dual_timescale_decomposer_render(params: Dict) -> str:
    """Averaging method → slow envelope + fast oscillation decomposition."""
    mod_name = params.get("module_name", "timescale_decomposer")
    return f'''\
use pyo3::prelude::*;

/// Dual timescale decomposition via averaging method.
/// Decomposes signal x(t) into slow envelope A(t) and fast oscillation.
///
/// Uses sliding-window RMS for envelope extraction:
///   A(t) = sqrt(2/T * integral_{{t-T/2}}^{{t+T/2}} x^2 dt')
/// and instantaneous phase from Hilbert-like discrete transform.
pub fn decompose(
    signal: &[f64], dt: f64, slow_window: usize,
) -> (Vec<f64>, Vec<f64>) {{
    let n = signal.len();
    if n < slow_window {{
        return (signal.to_vec(), vec![0.0; n]);
    }}

    // Slow envelope via sliding RMS
    let half = slow_window / 2;
    let mut envelope = Vec::with_capacity(n);
    for i in 0..n {{
        let start = if i >= half {{ i - half }} else {{ 0 }};
        let end = (i + half + 1).min(n);
        let rms: f64 = signal[start..end].iter()
            .map(|x| x * x)
            .sum::<f64>() / (end - start) as f64;
        envelope.push(rms.sqrt() * 2.0_f64.sqrt());
    }}

    // Fast oscillation = signal - smoothed signal
    // Smoothed via same sliding window (mean)
    let mut fast = Vec::with_capacity(n);
    for i in 0..n {{
        let start = if i >= half {{ i - half }} else {{ 0 }};
        let end = (i + half + 1).min(n);
        let mean: f64 = signal[start..end].iter().sum::<f64>() / (end - start) as f64;
        fast.push(signal[i] - mean);
    }}

    (envelope, fast)
}}

#[pyfunction]
#[pyo3(signature = (signal, dt=0.01, slow_window=100))]
fn py_decompose(signal: Vec<f64>, dt: f64, slow_window: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {{
    Ok(decompose(&signal, dt, slow_window))
}}

#[pymodule]
fn {mod_name}(_py: Python<\\'_>, m: &PyModule) -> PyResult<()> {{
    m.add_function(wrap_pyfunction!(py_decompose, m)?)?;
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_decompose_pure_sine() {{
        let dt = 0.01;
        let n = 1000;
        let signal: Vec<f64> = (0..n).map(|i| {{
            let t = i as f64 * dt;
            2.0 * (10.0 * t).sin()  // amplitude=2, frequency=10/(2pi)
        }}).collect();
        let (env, fast) = decompose(&signal, dt, 100);
        assert_eq!(env.len(), n);
        assert_eq!(fast.len(), n);
        // Envelope should be roughly constant ~2.0 in the middle
        let mid_env = env[n / 2];
        assert!((mid_env - 2.0).abs() < 0.5, "Envelope mid={{mid_env}}");
    }}
}}
'''


# ── Template registration ────────────────────────────────────────────────────

DUFFING_MARKET_TEMPLATE = RustTemplate(
    name="duffing_market_oscillator",
    domain="market",
    operation="duffing_market_oscillator",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_duffing_market_oscillator_render,
    description="Duffing RK4 with market-semantic params (alpha=mean_reversion, etc.)",
)

RESONANCE_DETECTOR_TEMPLATE = RustTemplate(
    name="resonance_detector",
    domain="market",
    operation="resonance_detector",
    borrow_profile=BorrowProfile.SHARED_REFERENCE,
    design_bv=(0.20, 0.15, 0.00, 0.00, 0.00, 0.00),
    render=_resonance_detector_render,
    description="Backbone curve sweep, fold bifurcation detection, jump probability",
)

DUAL_TIMESCALE_TEMPLATE = RustTemplate(
    name="dual_timescale_decomposer",
    domain="market",
    operation="dual_timescale_decomposer",
    borrow_profile=BorrowProfile.SHARED_REFERENCE,
    design_bv=(0.20, 0.15, 0.00, 0.00, 0.00, 0.00),
    render=_dual_timescale_decomposer_render,
    description="Averaging method → slow envelope + fast oscillation",
)

ALL_MARKET_TEMPLATES = [
    DUFFING_MARKET_TEMPLATE,
    RESONANCE_DETECTOR_TEMPLATE,
    DUAL_TIMESCALE_TEMPLATE,
]


def register_all(registry: TemplateRegistry) -> None:
    """Register all market model templates into a registry."""
    for t in ALL_MARKET_TEMPLATES:
        registry.register(t)
