"""
HDVS Navigator — 3-mode state machine for navigating High-Dimensional Vector Space.

Mathematical basis (LOGIC_FLOW.md, Section 0F):
  Three modes corresponding to different geometric regions:
    MODE_LCA:        exponential-dominant patch (ρ < ε₁, commutator < δ₁)
    MODE_TRANSITION: intermediate region (monitoring both metrics)
    MODE_KOOPMAN:    nonlinear corridor (spectral tracking via EDMD)

  Transitions:
    LCA → TRANSITION:   ρ > ε₁ OR commutator > δ₁
    TRANSITION → KOOPMAN: ρ > ε₂ AND spectral gap Δ > γ AND eigenfunctions stable
    KOOPMAN → LCA:      ρ < ε₁ AND commutator < δ₁ AND Koopman eigs → exponential
    TRANSITION → LCA:   ρ < ε₁ AND commutator < δ₁

Reference: LOGIC_FLOW.md Sections 0D, 0F
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any

from tensor.lca_patch_detector import LCAPatchDetector, PatchClassification
from tensor.koopman_edmd import EDMDKoopman, KoopmanResult
from tensor.bifurcation_detector import BifurcationDetector, BifurcationStatus


# Mode constants
MODE_LCA = 'lca'
MODE_TRANSITION = 'transition'
MODE_KOOPMAN = 'koopman'


@dataclass
class NavigationStep:
    """One step of navigation history."""
    mode: str                          # 'lca' | 'transition' | 'koopman'
    curvature_ratio: float             # ρ(x) at this step
    commutator_norm: float             # max ‖[Aᵢ,Aⱼ]‖_F
    spectral_gap: float                # Koopman or Jacobian spectral gap
    bifurcation_status: str            # 'stable' | 'critical' | 'bifurcation'
    operator_rank: int                 # intrinsic operator dimension
    patch_type: str                    # 'lca' | 'nonabelian' | 'chaotic'
    active_basis: Optional[np.ndarray] = None  # operator basis for current mode


@dataclass
class NavigatorThresholds:
    """Threshold parameters for mode switching."""
    eps1: float = 0.05    # LCA → TRANSITION curvature threshold
    eps2: float = 0.15    # TRANSITION → KOOPMAN curvature threshold
    delta1: float = 0.01  # commutator norm threshold for LCA
    gamma: float = 0.1    # minimum Koopman spectral gap
    stability_tol: float = 0.1  # eigenfunction stability tolerance


class HDVSNavigator:
    """
    Runtime navigation controller for the 3-mode HDVS state machine.

    Implements the state machine from LOGIC_FLOW.md Section 0F:
      LCA ↔ TRANSITION ↔ KOOPMAN

    Usage:
        def f(x): ...  # system vector field
        lca_detector = LCAPatchDetector(f, n_states=2)
        koopman = EDMDKoopman(observable_degree=2)
        navigator = HDVSNavigator(lca_detector, koopman)

        for x in trajectory:
            mode = navigator.step(x, x_prev)
            basis = navigator.get_active_basis()
    """

    def __init__(
        self,
        lca_detector: LCAPatchDetector,
        koopman: EDMDKoopman,
        thresholds: Optional[NavigatorThresholds] = None,
        window_size: int = 10,
    ):
        """
        Args:
            lca_detector: LCAPatchDetector for patch classification
            koopman: EDMDKoopman for spectral tracking in nonlinear regime
            thresholds: mode-switching thresholds (default: Section 0F values)
            window_size: number of samples used for region classification
        """
        self.detector = lca_detector
        self.koopman = koopman
        self.thresholds = thresholds or NavigatorThresholds()
        self.window_size = window_size

        # Internal state
        self._mode: str = MODE_LCA
        self._history: List[NavigationStep] = []
        self._recent_states: List[np.ndarray] = []
        self._recent_pairs: List[tuple] = []
        self._prev_koopman: Optional[KoopmanResult] = None
        self._active_basis: Optional[np.ndarray] = None
        self._bif_detector = BifurcationDetector(zero_tol=thresholds.eps1 if thresholds else 0.05)
        self._mode_hold_count: int = 0  # steps since last mode change

    # ------------------------------------------------------------------
    # Main step method
    # ------------------------------------------------------------------

    def step(
        self,
        x_current: np.ndarray,
        x_prev: Optional[np.ndarray] = None,
    ) -> str:
        """
        Process one state sample and return the current navigation mode.

        Args:
            x_current: current state vector (n,)
            x_prev: previous state vector (n,), needed for Koopman pairs

        Returns:
            mode: 'lca' | 'transition' | 'koopman'
        """
        # Accumulate recent states for windowed classification
        self._recent_states.append(x_current.copy())
        if x_prev is not None:
            self._recent_pairs.append((x_prev.copy(), x_current.copy()))
        if len(self._recent_states) > self.window_size:
            self._recent_states.pop(0)
        if len(self._recent_pairs) > self.window_size:
            self._recent_pairs.pop(0)

        # Need enough samples for classification
        if len(self._recent_states) < 3:
            self._history.append(self._make_step(
                mode=self._mode,
                curvature_ratio=0.0,
                commutator_norm=0.0,
                spectral_gap=0.0,
                bif_status='stable',
                operator_rank=1,
                patch_type='lca',
            ))
            return self._mode

        # Compute metrics
        x_arr = np.array(self._recent_states)
        classification = self.detector.classify_region(x_arr)
        rho = classification.curvature_ratio
        comm = classification.commutator_norm

        # Bifurcation check using Jacobian eigenvalues
        eigvals = classification.eigenvalues
        bif_result = self._bif_detector.check(eigvals)

        # Koopman spectral metrics (if we have pairs)
        koopman_gap = 0.0
        koopman_stable = False
        if len(self._recent_pairs) >= 5:
            try:
                self.koopman.fit(self._recent_pairs)
                curr_koopman = self.koopman.eigendecomposition()
                koopman_gap = curr_koopman.spectral_gap
                stability = self.koopman.eigenfunction_stability(
                    prev_result=self._prev_koopman,
                    curr_result=curr_koopman,
                )
                koopman_stable = stability < self.thresholds.stability_tol
                self._prev_koopman = curr_koopman
            except Exception:
                pass  # Koopman may fail on degenerate trajectories

        # Determine active basis
        self._active_basis = classification.basis_matrices

        # Mode transitions
        new_mode = self._transition(
            rho=rho,
            comm=comm,
            koopman_gap=koopman_gap,
            koopman_stable=koopman_stable,
        )

        # Record step
        self._history.append(self._make_step(
            mode=new_mode,
            curvature_ratio=rho,
            commutator_norm=comm,
            spectral_gap=koopman_gap if new_mode == MODE_KOOPMAN else classification.spectral_gap,
            bif_status=bif_result.status,
            operator_rank=classification.operator_rank,
            patch_type=classification.patch_type,
        ))

        self._mode = new_mode
        return new_mode

    # ------------------------------------------------------------------
    # State machine transition logic
    # ------------------------------------------------------------------

    def _transition(
        self,
        rho: float,
        comm: float,
        koopman_gap: float,
        koopman_stable: bool,
    ) -> str:
        """Apply 3-mode state machine transitions."""
        t = self.thresholds

        if self._mode == MODE_LCA:
            if rho > t.eps1 or comm > t.delta1:
                return MODE_TRANSITION
            return MODE_LCA

        elif self._mode == MODE_TRANSITION:
            if rho < t.eps1 and comm < t.delta1:
                return MODE_LCA
            if rho > t.eps2 and koopman_gap > t.gamma and koopman_stable:
                return MODE_KOOPMAN
            return MODE_TRANSITION

        elif self._mode == MODE_KOOPMAN:
            if rho < t.eps1 and comm < t.delta1:
                return MODE_LCA
            return MODE_KOOPMAN

        return self._mode  # fallback (shouldn't happen)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_active_basis(self) -> Optional[np.ndarray]:
        """Return operator basis for current mode (r × n × n)."""
        return self._active_basis

    def current_mode(self) -> str:
        """Return current navigation mode."""
        return self._mode

    def navigation_history(self) -> List[NavigationStep]:
        """Return full navigation history."""
        return list(self._history)

    def mode_sequence(self) -> List[str]:
        """Return just the mode at each step."""
        return [step.mode for step in self._history]

    def reset(self) -> None:
        """Reset navigator to initial state."""
        self._mode = MODE_LCA
        self._history.clear()
        self._recent_states.clear()
        self._recent_pairs.clear()
        self._prev_koopman = None
        self._active_basis = None
        self._bif_detector.reset()
        self._mode_hold_count = 0

    def summary(self) -> Dict[str, Any]:
        """Return mode statistics over navigation history."""
        if not self._history:
            return {'n_steps': 0, 'mode_counts': {}, 'mode_fractions': {}}

        mode_counts: Dict[str, int] = {}
        for step in self._history:
            mode_counts[step.mode] = mode_counts.get(step.mode, 0) + 1

        n = len(self._history)
        mode_fractions = {m: c / n for m, c in mode_counts.items()}

        return {
            'n_steps': n,
            'mode_counts': mode_counts,
            'mode_fractions': mode_fractions,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_step(
        self,
        mode: str,
        curvature_ratio: float,
        commutator_norm: float,
        spectral_gap: float,
        bif_status: str,
        operator_rank: int,
        patch_type: str,
    ) -> NavigationStep:
        return NavigationStep(
            mode=mode,
            curvature_ratio=curvature_ratio,
            commutator_norm=commutator_norm,
            spectral_gap=spectral_gap,
            bifurcation_status=bif_status,
            operator_rank=operator_rank,
            patch_type=patch_type,
            active_basis=self._active_basis,
        )
