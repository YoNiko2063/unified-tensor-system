"""
GET /api/v1/regime/status
Uses LCAPatchDetector on a synthetic 2-D LCA reference trajectory.
Returns the latest patch classification for monitoring.
"""
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from tensor.lca_patch_detector import LCAPatchDetector

router = APIRouter()


def _build_detector() -> LCAPatchDetector:
    """Construct a detector around a stable linear system (LCA reference)."""
    A = np.array([[-0.15, 1.0], [-1.0, -0.15]])
    fn = lambda x: A.dot(x)
    return LCAPatchDetector(fn, n_states=2, eps_curvature=0.05, delta_commutator=0.01)


def _synthetic_trajectory(n_steps: int = 200, dt: float = 0.05) -> np.ndarray:
    A = np.array([[-0.15, 1.0], [-1.0, -0.15]])
    x = np.array([1.0, 0.0])
    traj = []
    for _ in range(n_steps):
        traj.append(x.copy())
        x = x + dt * A.dot(x)
    return np.array(traj)


class RegimeStatus(BaseModel):
    patch_type: str
    commutator_norm: float
    curvature_ratio: float
    koopman_trust: float
    spectral_gap: float


@router.get("/status", response_model=RegimeStatus)
def get_regime_status() -> RegimeStatus:
    """Return current LCA / nonabelian / chaotic classification on a synthetic state."""
    detector = _build_detector()
    traj = _synthetic_trajectory()
    results = detector.classify_trajectory(traj, window=10)

    # Aggregate: take the median of numeric metrics, majority-vote patch_type
    from collections import Counter
    patch_votes = Counter(r.patch_type for r in results)
    dominant_patch = patch_votes.most_common(1)[0][0]

    commutator_norms = [r.commutator_norm for r in results]
    curvature_ratios = [r.curvature_ratio for r in results]
    koopman_trusts = [r.koopman_trust for r in results]
    spectral_gaps = [r.spectral_gap for r in results]

    return RegimeStatus(
        patch_type=dominant_patch,
        commutator_norm=float(np.median(commutator_norms)),
        curvature_ratio=float(np.median(curvature_ratios)),
        koopman_trust=float(np.median(koopman_trusts)),
        spectral_gap=float(np.median(spectral_gaps)),
    )
