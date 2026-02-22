"""Trading simulation templates — multi-asset price dynamics as compiled Rust kernels.

Maps the universal trading equation onto Rust:
  C·ṗ = u - G·p - h(p)

where:
  C = blended inertia matrix from agent classes
  G = transaction cost dissipation
  h(p) = amplitude * tanh(p / threshold)  (crowd behavior)
  u(t) = external signal (news, sentiment)

Templates:
  multi_asset_price_sim      — Euler integration of K-asset price dynamics
  agent_response_kernel      — blend K agent classes into composite C matrix
  crowd_behavior_kernel      — nonlinear h(p) = amplitude * tanh(p / threshold)
  regime_transition_detector — detect eigenvalue real-part sign changes

All templates: no ndarray crate, manual loops only.
"""

from __future__ import annotations

from typing import Dict

from codegen.intent_spec import BorrowProfile
from codegen.template_registry import RustTemplate, TemplateRegistry


# ── Render functions ────────────────────────────────────────────────────────

def _multi_asset_price_sim_render(params: Dict) -> str:
    """Euler integration of K-asset price dynamics: C·ṗ = u - G·p - h(p).

    Pure functional on owned Vec<f64> state.
    No ndarray — flat manual indexing.
    """
    n_assets = int(params.get("n_assets", 3))
    dt = float(params.get("dt", 0.01))
    n_steps = int(params.get("n_steps", 100))

    return f'''\
/// Multi-asset price dynamics: C·ṗ = u - G·p - h(p)
/// Euler integration, flat manual matrix multiply, no ndarray.
///
/// State layout: prices[0..{n_assets}]
/// C, G: {n_assets}×{n_assets} row-major flat f64 arrays
/// h(p) = amplitude * tanh(p / threshold)

pub fn multi_asset_price_sim(
    prices_init: Vec<f64>,
    c_matrix: Vec<f64>,
    g_matrix: Vec<f64>,
    u_signal: Vec<f64>,
    amplitude: f64,
    threshold: f64,
) -> Vec<Vec<f64>> {{
    let n: usize = {n_assets};
    let dt: f64 = {dt};
    let n_steps: usize = {n_steps};

    debug_assert_eq!(prices_init.len(), n, "prices must have length {n_assets}");
    debug_assert_eq!(c_matrix.len(), n * n, "C must be {n_assets}×{n_assets} flat");
    debug_assert_eq!(g_matrix.len(), n * n, "G must be {n_assets}×{n_assets} flat");
    debug_assert_eq!(u_signal.len(), n, "u must have length {n_assets}");

    let mut prices = prices_init;
    let mut trajectory: Vec<Vec<f64>> = Vec::with_capacity(n_steps + 1);
    trajectory.push(prices.clone());

    for _ in 0..n_steps {{
        // Compute G·p  (flat row-major mat-vec)
        let mut gp = vec![0.0_f64; n];
        for i in 0..n {{
            let mut acc = 0.0_f64;
            for j in 0..n {{
                acc += g_matrix[i * n + j] * prices[j];
            }}
            gp[i] = acc;
        }}

        // Compute h(p) = amplitude * tanh(p / threshold)
        let mut hp = vec![0.0_f64; n];
        for i in 0..n {{
            hp[i] = amplitude * (prices[i] / threshold).tanh();
        }}

        // RHS = u - G·p - h(p)
        let mut rhs = vec![0.0_f64; n];
        for i in 0..n {{
            rhs[i] = u_signal[i] - gp[i] - hp[i];
        }}

        // Solve C·ṗ = rhs  →  ṗ = C⁻¹·rhs
        // Approximate via diagonal of C (simplification for kernel use)
        let mut dp = vec![0.0_f64; n];
        for i in 0..n {{
            let c_ii = c_matrix[i * n + i];
            dp[i] = if c_ii.abs() > 1e-15 {{ rhs[i] / c_ii }} else {{ 0.0 }};
        }}

        // Euler step
        for i in 0..n {{
            prices[i] += dt * dp[i];
        }}
        trajectory.push(prices.clone());
    }}

    trajectory
}}

#[cfg(test)]
mod tests_multi_asset {{
    use super::*;

    #[test]
    fn test_trajectory_length() {{
        let n = {n_assets};
        let prices_init = vec![100.0_f64; n];
        let c = vec![1.0_f64; n * n];  // ones matrix
        let mut c_diag = vec![0.0_f64; n * n];
        for i in 0..n {{ c_diag[i * n + i] = 1.0; }}
        let g = vec![0.0_f64; n * n];
        let u = vec![0.0_f64; n];
        let traj = multi_asset_price_sim(prices_init, c_diag, g, u, 0.1, 10.0);
        assert_eq!(traj.len(), {n_steps} + 1);
    }}

    #[test]
    fn test_tanh_crowd_effect() {{
        // With large u and amplitude > 0 → crowd pushes prices
        let n = {n_assets};
        let mut c_diag = vec![0.0_f64; n * n];
        for i in 0..n {{ c_diag[i * n + i] = 1.0; }}
        let g = vec![0.0_f64; n * n];
        let u = vec![1.0_f64; n];  // constant input
        let traj = multi_asset_price_sim(vec![0.0_f64; n], c_diag, g, u, 1.0, 10.0);
        // Prices should increase over time with positive u
        let final_price = traj.last().unwrap()[0];
        assert!(final_price > 0.0, "Prices should rise: {{final_price}}");
    }}
}}
'''


def _agent_response_kernel_render(params: Dict) -> str:
    """Blend K agent classes into composite inertia matrix C.

    C = Σ_k w_k * diag(1/response_time_k)

    Shared reference to agent parameter slice.
    BV = (0.20, 0.10, 0.00, 0.00, 0.00, 0.00)
    """
    n_agents = int(params.get("n_agents", 3))
    n_assets = int(params.get("n_assets", 3))

    return f'''\
/// Blend {n_agents} agent classes into composite inertia matrix C.
/// C = Σ_k w_k * diag(1 / response_time_k)
///
/// agent_params layout per agent: [weight, response_time]  (2 floats)
/// output c_matrix: {n_assets}×{n_assets} flat row-major, diagonal blending

pub fn agent_response_kernel(
    agent_params: &[f64],
    n_agents: usize,
    n_assets: usize,
) -> Vec<f64> {{
    debug_assert_eq!(
        agent_params.len(), n_agents * 2,
        "agent_params must be n_agents × 2 [weight, response_time]"
    );

    let mut c_matrix = vec![0.0_f64; n_assets * n_assets];

    for k in 0..n_agents {{
        let weight = agent_params[k * 2];
        let response_time = agent_params[k * 2 + 1];
        let inv_rt = if response_time.abs() > 1e-15 {{
            1.0 / response_time
        }} else {{
            0.0
        }};
        // Add w_k * diag(1/response_time_k) to C
        for i in 0..n_assets {{
            c_matrix[i * n_assets + i] += weight * inv_rt;
        }}
    }}

    c_matrix
}}

/// Normalise agent weights so they sum to 1.
pub fn normalise_agent_weights(agent_params: &mut Vec<f64>, n_agents: usize) {{
    let total_weight: f64 = (0..n_agents).map(|k| agent_params[k * 2]).sum();
    if total_weight.abs() > 1e-15 {{
        for k in 0..n_agents {{
            agent_params[k * 2] /= total_weight;
        }}
    }}
}}

#[cfg(test)]
mod tests_agent_kernel {{
    use super::*;

    #[test]
    fn test_single_agent_diagonal() {{
        // One agent, weight=1, response_time=2 → C_ii = 0.5
        let params = vec![1.0_f64, 2.0];
        let c = agent_response_kernel(&params, 1, {n_assets});
        for i in 0..{n_assets} {{
            assert!((c[i * {n_assets} + i] - 0.5).abs() < 1e-12,
                    "Diagonal mismatch at {{i}}: {{}}", c[i * {n_assets} + i]);
        }}
    }}

    #[test]
    fn test_multiple_agents_blend() {{
        // Two agents: w=0.5, rt=1 and w=0.5, rt=2
        // Expected: 0.5*1.0 + 0.5*0.5 = 0.75 on diagonal
        let params = vec![0.5_f64, 1.0, 0.5, 2.0];
        let c = agent_response_kernel(&params, 2, {n_assets});
        let expected = 0.5 * 1.0 + 0.5 * 0.5;  // 0.75
        for i in 0..{n_assets} {{
            assert!((c[i * {n_assets} + i] - expected).abs() < 1e-12,
                    "Blend mismatch: {{}}", c[i * {n_assets} + i]);
        }}
    }}
}}
'''


def _crowd_behavior_kernel_render(params: Dict) -> str:
    """Nonlinear crowd behavior: h(p) = amplitude * tanh(p / threshold).

    Pure functional — owned Vec<f64> in, owned Vec<f64> out.
    BV = (0.10, 0.00, 0.00, 0.00, 0.00, 0.00)
    """
    amplitude = float(params.get("amplitude", 1.0))
    threshold = float(params.get("threshold", 10.0))

    return f'''\
/// Crowd behavior nonlinearity: h(p) = amplitude * tanh(p / threshold)
///
/// Models herd dynamics with tanh saturation.
/// amplitude = {amplitude:.6f}
/// threshold = {threshold:.6f}

pub fn crowd_behavior_kernel(prices: Vec<f64>) -> Vec<f64> {{
    let amplitude: f64 = {amplitude:.6f};
    let threshold: f64 = {threshold:.6f};

    prices
        .into_iter()
        .map(|p| amplitude * (p / threshold).tanh())
        .collect()
}}

/// Jacobian of h w.r.t. p (diagonal, since h is element-wise):
/// dh_i/dp_i = amplitude / threshold * sech²(p_i / threshold)
pub fn crowd_behavior_jacobian(prices: &[f64]) -> Vec<f64> {{
    let amplitude: f64 = {amplitude:.6f};
    let threshold: f64 = {threshold:.6f};

    prices
        .iter()
        .map(|&p| {{
            let t = (p / threshold).tanh();
            amplitude / threshold * (1.0 - t * t)  // sech² = 1 - tanh²
        }})
        .collect()
}}

#[cfg(test)]
mod tests_crowd_kernel {{
    use super::*;

    #[test]
    fn test_zero_price_gives_zero() {{
        let h = crowd_behavior_kernel(vec![0.0, 0.0, 0.0]);
        for v in &h {{
            assert!(v.abs() < 1e-15, "h(0) must be 0, got {{v}}");
        }}
    }}

    #[test]
    fn test_saturation_at_large_price() {{
        // For p >> threshold: tanh(p/threshold) → 1 → h → amplitude
        let amplitude = {amplitude:.6f};
        let threshold = {threshold:.6f};
        let large_p = threshold * 100.0;
        let h = crowd_behavior_kernel(vec![large_p]);
        assert!((h[0] - amplitude).abs() < 1e-6,
                "Expected saturation at amplitude={{}}", amplitude);
    }}

    #[test]
    fn test_antisymmetry() {{
        // h(-p) = -h(p) because tanh is odd
        let h_pos = crowd_behavior_kernel(vec![5.0_f64]);
        let h_neg = crowd_behavior_kernel(vec![-5.0_f64]);
        assert!((h_pos[0] + h_neg[0]).abs() < 1e-12,
                "tanh must be antisymmetric");
    }}

    #[test]
    fn test_jacobian_positive() {{
        // Jacobian of tanh is always positive
        let jac = crowd_behavior_jacobian(&[0.0, 1.0, -1.0, 100.0]);
        for &v in &jac {{
            assert!(v >= 0.0, "Jacobian must be non-negative: {{v}}");
        }}
    }}
}}
'''


def _regime_transition_detector_render(params: Dict) -> str:
    """Detect when eigenvalue real part crosses zero (stability boundary).

    Mutable output flag + shared reference to eigenvalue array.
    BV = (0.20, 0.10, 0.00, 0.20, 0.00, 0.00)
    """
    n_eigenvalues = int(params.get("n_eigenvalues", 4))
    stability_threshold = float(params.get("stability_threshold", 0.0))

    return f'''\
/// Detect regime transitions: scan eigenvalue real parts for sign changes.
///
/// Returns (crossed: bool, index: usize) — whether any Re(λ) crossed the
/// stability boundary ({stability_threshold:.6f}) and the index of the first crossing.
///
/// eigenvalues layout: flat [re_0, im_0, re_1, im_1, ...] (interleaved)

pub fn regime_transition_detector(
    eigenvalues: &[f64],
    prev_signs: &mut Vec<i8>,
    crossed: &mut bool,
    first_crossing_index: &mut usize,
) {{
    let n = {n_eigenvalues};
    let threshold: f64 = {stability_threshold:.6f};

    debug_assert_eq!(eigenvalues.len(), n * 2, "eigenvalues must be n×2 interleaved");

    // Initialise prev_signs on first call
    if prev_signs.is_empty() {{
        prev_signs.resize(n, 0i8);
        for k in 0..n {{
            let re = eigenvalues[k * 2];
            prev_signs[k] = if re > threshold {{ 1i8 }} else {{ -1i8 }};
        }}
        *crossed = false;
        *first_crossing_index = usize::MAX;
        return;
    }}

    *crossed = false;
    *first_crossing_index = usize::MAX;

    for k in 0..n {{
        let re = eigenvalues[k * 2];
        let sign_now: i8 = if re > threshold {{ 1 }} else {{ -1 }};
        if sign_now != prev_signs[k] {{
            // Sign flip detected — stability boundary crossed
            if !*crossed {{
                *first_crossing_index = k;
            }}
            *crossed = true;
            prev_signs[k] = sign_now;
        }}
    }}
}}

/// Convenience wrapper: returns (crossed, first_index) without external state.
/// Uses a copy of prev_signs passed by value.
pub fn detect_stability_crossing(
    eigenvalues: &[f64],
    prev_eigenvalues: &[f64],
) -> (bool, usize) {{
    let n = {n_eigenvalues};
    let threshold: f64 = {stability_threshold:.6f};

    if eigenvalues.len() < n * 2 || prev_eigenvalues.len() < n * 2 {{
        return (false, usize::MAX);
    }}

    let mut crossed = false;
    let mut first_idx = usize::MAX;

    for k in 0..n {{
        let re_prev = prev_eigenvalues[k * 2];
        let re_now  = eigenvalues[k * 2];
        let sign_prev: i8 = if re_prev > threshold {{ 1 }} else {{ -1 }};
        let sign_now:  i8 = if re_now  > threshold {{ 1 }} else {{ -1 }};
        if sign_prev != sign_now {{
            if !crossed {{
                first_idx = k;
            }}
            crossed = true;
        }}
    }}

    (crossed, first_idx)
}}

#[cfg(test)]
mod tests_regime_detector {{
    use super::*;

    #[test]
    fn test_no_crossing_stable() {{
        // All eigenvalues with negative real parts → no crossing
        let eigs = vec![-0.1_f64, 0.5, -0.2, -0.3, -0.05, 0.1, -0.3, 0.2];
        let prev = vec![-0.1_f64, 0.5, -0.2, -0.3, -0.05, 0.1, -0.3, 0.2];
        let (crossed, idx) = detect_stability_crossing(&eigs, &prev);
        assert!(!crossed, "No sign change should not trigger crossing");
    }}

    #[test]
    fn test_crossing_detected() {{
        // One eigenvalue flips from negative to positive real part
        let prev = vec![-0.1_f64, 0.0, -0.2, 0.0, -0.05, 0.0, -0.3, 0.0];
        let curr = vec![ 0.1_f64, 0.0, -0.2, 0.0, -0.05, 0.0, -0.3, 0.0];
        let (crossed, idx) = detect_stability_crossing(&curr, &prev);
        assert!(crossed, "Sign change in first eigenvalue should trigger crossing");
        assert_eq!(idx, 0);
    }}

    #[test]
    fn test_first_crossing_index() {{
        // Two crossings — should report index of first one
        let prev = vec![-0.1_f64, 0.0, -0.2, 0.0, -0.05, 0.0, -0.3, 0.0];
        let curr = vec![-0.1_f64, 0.0,  0.2, 0.0,  0.05, 0.0, -0.3, 0.0];
        let (crossed, idx) = detect_stability_crossing(&curr, &prev);
        assert!(crossed);
        assert_eq!(idx, 1, "Second eigenvalue crossed first (at index 1)");
    }}

    #[test]
    fn test_stateful_detector_initialises() {{
        let eigs = vec![-0.1_f64, 0.0, -0.2, 0.0, -0.05, 0.0, -0.3, 0.0];
        let mut prev_signs: Vec<i8> = Vec::new();
        let mut crossed = false;
        let mut first_idx: usize = 0;
        // First call initialises state — should not report crossing
        regime_transition_detector(&eigs, &mut prev_signs, &mut crossed, &mut first_idx);
        assert!(!crossed, "Initialisation call must not report crossing");
    }}
}}
'''


# ── Template registration ────────────────────────────────────────────────────

MULTI_ASSET_PRICE_SIM_TEMPLATE = RustTemplate(
    name="multi_asset_price_sim",
    domain="trading",
    operation="multi_asset_price_sim",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_multi_asset_price_sim_render,
    requires_cargo=False,
    description="Euler integration of K-asset price dynamics via C·ṗ = u - G·p - h(p). "
                "Pure functional, no ndarray, manual loops.",
)

AGENT_RESPONSE_KERNEL_TEMPLATE = RustTemplate(
    name="agent_response_kernel",
    domain="trading",
    operation="agent_response_kernel",
    borrow_profile=BorrowProfile.SHARED_REFERENCE,
    design_bv=(0.20, 0.10, 0.00, 0.00, 0.00, 0.00),
    render=_agent_response_kernel_render,
    requires_cargo=False,
    description="Blend K agent classes into composite inertia matrix C = Σ_k w_k * diag(1/rt_k). "
                "Shared reference to agent parameter slice.",
)

CROWD_BEHAVIOR_KERNEL_TEMPLATE = RustTemplate(
    name="crowd_behavior_kernel",
    domain="trading",
    operation="crowd_behavior_kernel",
    borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
    design_bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    render=_crowd_behavior_kernel_render,
    requires_cargo=False,
    description="Nonlinear crowd behavior h(p) = amplitude * tanh(p / threshold). Pure functional.",
)

REGIME_TRANSITION_DETECTOR_TEMPLATE = RustTemplate(
    name="regime_transition_detector",
    domain="trading",
    operation="regime_transition_detector",
    borrow_profile=BorrowProfile.MUTABLE_OUTPUT,
    design_bv=(0.20, 0.10, 0.00, 0.20, 0.00, 0.00),
    render=_regime_transition_detector_render,
    requires_cargo=False,
    description="Detect eigenvalue real-part sign changes (stability boundary crossings). "
                "Mutable output flag + shared reference to eigenvalue array.",
)

ALL_TRADING_TEMPLATES = [
    MULTI_ASSET_PRICE_SIM_TEMPLATE,
    AGENT_RESPONSE_KERNEL_TEMPLATE,
    CROWD_BEHAVIOR_KERNEL_TEMPLATE,
    REGIME_TRANSITION_DETECTOR_TEMPLATE,
]


def register_all(registry: TemplateRegistry) -> None:
    """Register all trading simulation templates into a registry."""
    for t in ALL_TRADING_TEMPLATES:
        registry.register(t)
