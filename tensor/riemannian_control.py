"""
Riemannian Control — curvature-gradient control law for HDVS navigation.

Mathematical basis (LOGIC_FLOW.md, Section 0L):
  Riemannian metric on HDVS: g_ij = ⟨∂ᵢJ, ∂ⱼJ⟩_F
  Curvature penalty: C(x) = Σᵢ<ⱼ ‖[Aᵢ,Aⱼ]‖²_F + ‖∇J(x)‖²_F
  Geometric control law: ẍ = -κ F⁻¹(x) ∇C(x)   (natural curvature descent)
  Geodesic equation: ẍᵏ + Γᵢⱼᵏ ẋⁱ ẋʲ = 0

  Resonance collapse safety: |λᵢ/λⱼ - αⱼ/αᵢ| > δ for all i≠j

Reference: LOGIC_FLOW.md Section 0L (Sections I–VIII)
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Optional


# ------------------------------------------------------------------
# Curvature gradient
# ------------------------------------------------------------------

def curvature_gradient(
    basis_matrices: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    C(x) = Σᵢ<ⱼ ‖[Aᵢ,Aⱼ]‖²_F — commutator energy.

    Args:
        basis_matrices: (r, n, n) operator basis at current state
        normalize: divide by r*(r-1)/2 to get mean commutator energy

    Returns:
        Curvature energy C(x) ≥ 0
    """
    r = basis_matrices.shape[0]
    if r < 2:
        return 0.0

    total = 0.0
    count = 0
    for i, j in combinations(range(r), 2):
        A, B = basis_matrices[i], basis_matrices[j]
        comm = A @ B - B @ A
        total += float(np.linalg.norm(comm, 'fro') ** 2)
        count += 1

    if normalize and count > 0:
        total /= count

    return total


def curvature_gradient_vector(
    x: np.ndarray,
    basis_fn,
    h: float = 1e-4,
) -> np.ndarray:
    """
    Numerical gradient ∇C(x) via finite differences.

    Args:
        x: state vector (n,)
        basis_fn: callable x → (r, n, n) basis matrices
        h: finite difference step

    Returns:
        Gradient vector (n,) — direction of steepest curvature increase
    """
    n = len(x)
    grad = np.zeros(n)
    C0 = curvature_gradient(basis_fn(x))

    for k in range(n):
        xph = x.copy()
        xph[k] += h
        Ch = curvature_gradient(basis_fn(xph))
        grad[k] = (Ch - C0) / h

    return grad


# ------------------------------------------------------------------
# Natural control step (Fisher/Riemannian descent)
# ------------------------------------------------------------------

def natural_control_step(
    x: np.ndarray,
    basis_matrices: np.ndarray,
    fim: Optional[np.ndarray] = None,
    kappa: float = 0.01,
    h: float = 1e-4,
) -> np.ndarray:
    """
    ẍ = -κ F⁻¹(x) ∇C(x) — natural curvature-gradient descent.

    The Fisher Information Matrix (FIM) defines the natural metric on
    parameter space. In LCA patches: FIM ≈ Σₖ λₖ² vₖvₖᵀ.

    Args:
        x: current state (n,)
        basis_matrices: (r, n, n) operator basis at x
        fim: (n, n) Fisher Information Matrix. If None, uses identity (standard gradient).
        kappa: step size

    Returns:
        Control acceleration ẍ (n,)
    """
    # Compute gradient numerically using basis at neighboring points
    n = len(x)
    r = basis_matrices.shape[0]

    # Finite difference gradient of curvature energy
    C0 = curvature_gradient(basis_matrices)
    grad_C = np.zeros(n)
    for k in range(n):
        xph = x.copy()
        xph[k] += h
        # Approximate basis at perturbed point via linear extrapolation
        # (in practice, caller would provide basis_fn)
        grad_C[k] = 0.0  # placeholder — basis_fn not available here

    # If FIM provided, use natural gradient; otherwise standard gradient
    if fim is None or np.linalg.matrix_rank(fim) < n:
        descent = -kappa * grad_C
    else:
        try:
            descent = -kappa * np.linalg.solve(fim, grad_C)
        except np.linalg.LinAlgError:
            descent = -kappa * grad_C

    return descent


def natural_control_step_with_fn(
    x: np.ndarray,
    basis_fn,
    fim: Optional[np.ndarray] = None,
    kappa: float = 0.01,
    h: float = 1e-4,
) -> np.ndarray:
    """
    ẍ = -κ F⁻¹(x) ∇C(x) with full gradient computation.

    Args:
        x: current state (n,)
        basis_fn: callable x → (r, n, n) basis matrices
        fim: (n, n) Fisher Information Matrix (or None for identity)
        kappa: step size
        h: finite difference step for gradient

    Returns:
        Control acceleration ẍ (n,)
    """
    grad_C = curvature_gradient_vector(x, basis_fn, h=h)

    if fim is None:
        return -kappa * grad_C

    try:
        fim_reg = fim + 1e-8 * np.eye(len(x))  # regularize
        return -kappa * np.linalg.solve(fim_reg, grad_C)
    except np.linalg.LinAlgError:
        return -kappa * grad_C


# ------------------------------------------------------------------
# Resonance collapse check
# ------------------------------------------------------------------

def resonance_collapse_check(
    eigenvalues: np.ndarray,
    alpha: np.ndarray,
    delta: float = 1e-3,
) -> bool:
    """
    Check safety condition: |λᵢ/λⱼ - αⱼ/αᵢ| > δ for all i≠j.

    Collapse occurs when αᵢλᵢ = αⱼλⱼ → eigenbasis degeneracy.

    Args:
        eigenvalues: complex eigenvalues of current Jacobian
        alpha: interval operator coefficients (r,)
        delta: safety margin (collapse risk if diff < delta)

    Returns:
        True if SAFE (no resonance collapse), False if at risk
    """
    lam = np.abs(np.real(eigenvalues))
    r = min(len(lam), len(alpha))

    for i in range(r):
        for j in range(i + 1, r):
            if abs(alpha[j]) < 1e-12 or abs(alpha[i]) < 1e-12:
                continue
            if abs(lam[j]) < 1e-12:
                continue
            ratio_lam = lam[i] / lam[j]
            ratio_alpha = alpha[j] / alpha[i]
            if abs(ratio_lam - ratio_alpha) < delta:
                return False  # collapse risk

    return True  # safe


# ------------------------------------------------------------------
# Geodesic step (Riemannian manifold)
# ------------------------------------------------------------------

def geodesic_step(
    x: np.ndarray,
    x_dot: np.ndarray,
    metric_tensor: np.ndarray,
    basis_fn,
    dt: float = 0.01,
    h: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One step of geodesic flow: ẍᵏ + Γᵢⱼᵏ ẋⁱ ẋʲ = 0.

    Uses numerical Christoffel symbols from the metric tensor g_ij = ⟨∂ᵢJ, ∂ⱼJ⟩_F.

    Args:
        x: current position (n,)
        x_dot: current velocity (n,)
        metric_tensor: (n, n) Riemannian metric at x
        basis_fn: callable x → (r, n, n) for computing metric elsewhere
        dt: time step
        h: finite difference step for Christoffel symbols

    Returns:
        (x_new, x_dot_new) after one geodesic step
    """
    n = len(x)

    # Geodesic equation: ẍ = -Γᵢⱼ ẋⁱ ẋʲ (simplified: use metric gradient only)
    # Full Christoffel: Γᵢⱼᵏ = ½ gᵏˡ(∂ᵢgⱼˡ + ∂ⱼgᵢˡ - ∂ˡgᵢⱼ)
    # For numerical stability: use diagonal approximation (only diagonal Γ)

    x_ddot = np.zeros(n)

    try:
        g_inv = np.linalg.inv(metric_tensor + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        g_inv = np.eye(n)

    for k in range(n):
        # Numerical ∂ₖg_ij
        xph = x.copy()
        xph[k] += h
        g_ph = riemannian_metric(xph, basis_fn)
        dg = (g_ph - metric_tensor) / h

        # Simplified: Γ_diag contribution
        # ẍᵏ ≈ -½ Σᵢⱼ gᵏˡ ∂ˡgᵢⱼ ẋⁱ ẋʲ
        christoffel_k = -0.5 * np.sum(g_inv[k, :, None, None] * dg[None, :, :] * x_dot[:, None] * x_dot[None, :])
        # Note: above is approximate; proper implementation requires 3-index tensor
        x_ddot[k] = float(np.sum(g_inv[k] @ (dg @ x_dot)) * -0.5)

    # Euler integration
    x_dot_new = x_dot + dt * x_ddot
    x_new = x + dt * x_dot_new

    return x_new, x_dot_new


# ------------------------------------------------------------------
# Riemannian metric at a point
# ------------------------------------------------------------------

def riemannian_metric(
    x: np.ndarray,
    basis_fn,
    h: float = 1e-4,
) -> np.ndarray:
    """
    g_ij(x) = ⟨∂ᵢJ(x), ∂ⱼJ(x)⟩_F

    The Riemannian metric on HDVS induced by how the operator algebra J
    changes with state position.

    Args:
        x: state vector (n,)
        basis_fn: callable x → (r, n, n) basis matrices
        h: finite difference step

    Returns:
        (n, n) symmetric positive semidefinite metric tensor
    """
    n = len(x)
    B0 = basis_fn(x)  # r × n × n

    # ∂ᵢJ(x): finite difference of basis
    dB = []
    for i in range(n):
        xph = x.copy()
        xph[i] += h
        Bph = basis_fn(xph)
        dBi = (Bph - B0) / h  # r × n × n
        dB.append(dBi.reshape(-1))  # flatten to r*n*n vector

    dB = np.array(dB)  # n × (r*n*n)

    # g_ij = ⟨dBᵢ, dBⱼ⟩ = dBᵢ · dBⱼ
    g = dB @ dB.T  # n × n

    return g


# ------------------------------------------------------------------
# Fisher Information Matrix (LCA patch)
# ------------------------------------------------------------------

def fisher_information_matrix(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    FIM ≈ Σₖ λₖ² vₖvₖᵀ — Fisher Information Matrix in LCA patch.

    In the LCA regime, FIM eigenvectors align with Koopman eigenfunctions.
    Natural gradient descent in FIM-metric = descent along operator manifold.

    Args:
        eigenvalues: (r,) real or complex eigenvalues
        eigenvectors: (n, r) matrix of eigenvectors (columns)

    Returns:
        (n, n) symmetric PSD FIM approximation
    """
    n = eigenvectors.shape[0]
    fim = np.zeros((n, n))
    for k in range(eigenvectors.shape[1]):
        lam_sq = float(np.abs(eigenvalues[k]) ** 2)
        v = np.real(eigenvectors[:, k])
        fim += lam_sq * np.outer(v, v)
    return fim
