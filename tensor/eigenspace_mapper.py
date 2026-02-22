"""
EigenspaceMapper — bridge from ecemath parameter space to HarmonicAtlas.

For any system C(θ)·ẋ + G(θ)·x + h(x,θ) = u(t), this module:

  1. Calls system_factory(θ) to build a concrete vector field
  2. Finds equilibrium x* via root-finding
  3. Generates state samples in an ε-ball around x*
  4. Classifies the regime via LCAPatchDetector (6-step pipeline)
  5. Registers PatchClassification in HarmonicAtlas
  6. Stores θ, domain, and derived spectral features in Patch.metadata

Patch.metadata schema:
  {
    'domain':          str,    # e.g. 'solar_mppt'
    'theta':           dict,   # physical parameter point
    'hurwitz_margin':  float,  # max Re(λ), negative ↔ Hurwitz stable
    'dominant_freq_hz': float, # |Im(λ_dom)| / (2π)
    'damping_ratio':   float,  # −Re(λ_dom) / |λ_dom|, ∈ [0,1]
  }

system_factory protocol (duck-typed, two accepted forms):
  Form A — DynamicalSystem: has .rhs(t, x) → ẋ,  .n_states() → int,
            and optionally .find_equilibrium(x0) → (x*, bool).
  Form B — plain vector field: callable f(x) → ẋ.  Requires n_states arg.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import root

from tensor.lca_patch_detector import LCAPatchDetector, PatchClassification
from tensor.harmonic_atlas import HarmonicAtlas
from tensor.patch_graph import Patch


# ── Spectral feature helpers ──────────────────────────────────────────────────

def hurwitz_margin(eigvals: np.ndarray) -> float:
    """max Re(λ). Negative ↔ Hurwitz stable. Zero ↔ marginal."""
    if len(eigvals) == 0:
        return 0.0
    return float(np.max(np.real(eigvals)))


def dominant_frequency_hz(eigvals: np.ndarray) -> float:
    """Dominant oscillatory frequency: max |Im(λ)| / (2π)."""
    if len(eigvals) == 0:
        return 0.0
    return float(np.max(np.abs(np.imag(eigvals)))) / (2.0 * np.pi)


def damping_ratio(eigvals: np.ndarray) -> float:
    """
    −Re(λ_dom) / |λ_dom| for the eigenvalue with largest |Im|.
    Returns 0.0 if |λ_dom| < 1e-12 (DC mode dominates).
    """
    if len(eigvals) == 0:
        return 0.0
    imag_mag = np.abs(np.imag(eigvals))
    idx = int(np.argmax(imag_mag))
    lam = eigvals[idx]
    mag = abs(lam)
    if mag < 1e-12:
        return 0.0
    return float(-np.real(lam) / mag)


# ── Equilibrium finder ────────────────────────────────────────────────────────

def _find_equilibrium(
    system_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Find x* where f(x*) = 0 via SciPy hybr method."""
    result = root(system_fn, x0, method='hybr')
    return result.x, bool(result.success)


# ── EigenspaceMapper ──────────────────────────────────────────────────────────

@dataclass
class MapResult:
    """Result of a single map_point call."""
    patch: Patch
    classification: PatchClassification
    theta: dict
    domain: str
    equilibrium: np.ndarray
    converged: bool


class EigenspaceMapper:
    """
    Maps physical parameter points θ to HarmonicAtlas patches.

    Usage (Form A — DynamicalSystem):
        def factory(theta):
            return MyDynamicalSystem(R=theta['R'], L=theta['L'], C=theta['C'])

        mapper = EigenspaceMapper(factory, atlas)
        result = mapper.map_point({'R': 50.0, 'L': 1e-3, 'C': 1e-6}, 'rlc_series')

    Usage (Form B — plain vector field):
        def factory(theta):
            R, L, C = theta['R'], theta['L'], theta['C']
            def f(x):
                v_C, i_L = x
                return np.array([i_L / C, (-v_C - R * i_L) / L])
            return f

        mapper = EigenspaceMapper(factory, atlas, n_states=2)
        result = mapper.map_point({'R': 50.0, 'L': 1e-3, 'C': 1e-6}, 'rlc_series')
    """

    def __init__(
        self,
        system_factory: Callable[[dict], Any],
        atlas: HarmonicAtlas,
        lca_detector: Optional[LCAPatchDetector] = None,
        n_states: Optional[int] = None,
        n_samples: int = 30,
        sample_radius: float = 0.01,
        rng_seed: Optional[int] = None,
    ):
        """
        Args:
            system_factory: theta -> DynamicalSystem or theta -> f(x)
            atlas:          HarmonicAtlas to register discovered patches
            lca_detector:   Optional template — threshold params (eps_curvature,
                            delta_commutator, rank_tol, h) are copied from it.
                            If None, defaults are used.
            n_states:       Required when system_factory returns a plain callable.
                            Ignored when it returns a DynamicalSystem.
            n_samples:      State samples generated around each equilibrium.
            sample_radius:  Gaussian σ for sample cloud around x*.
            rng_seed:       For reproducible sample generation.
        """
        self.system_factory = system_factory
        self.atlas = atlas
        self.n_states = n_states
        self.n_samples = n_samples
        self.sample_radius = sample_radius
        self._rng = np.random.default_rng(rng_seed)

        # Copy threshold params from template detector or use defaults
        if lca_detector is not None:
            self._eps_curvature = lca_detector.eps_curvature
            self._delta_commutator = lca_detector.delta_commutator
            self._rank_tol = lca_detector.rank_tol
            self._h = lca_detector.h
        else:
            self._eps_curvature = 0.05
            self._delta_commutator = 0.01
            self._rank_tol = 1e-2
            self._h = 1e-5

    # ── Core mapping ──────────────────────────────────────────────────────────

    def map_point(
        self,
        theta: dict,
        domain: str,
        x0: Optional[np.ndarray] = None,
    ) -> MapResult:
        """
        Map a single parameter point θ to an atlas patch.

        Args:
            theta:  physical parameter dict, e.g. {'R': 50.0, 'L': 1e-3}
            domain: human-readable domain label, e.g. 'solar_mppt'
            x0:     initial guess for equilibrium search.
                    Defaults to zeros(n_states).

        Returns:
            MapResult with patch, classification, equilibrium, convergence flag.
        """
        system_fn, n = self._unwrap(theta)
        x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)

        x_eq, converged = _find_equilibrium(system_fn, x0)

        # Sample cloud around equilibrium
        noise = self._rng.standard_normal((self.n_samples, n)) * self.sample_radius
        x_samples = x_eq[None, :] + noise  # (n_samples, n)

        # Classify regime
        detector = LCAPatchDetector(
            system_fn=system_fn,
            n_states=n,
            eps_curvature=self._eps_curvature,
            delta_commutator=self._delta_commutator,
            rank_tol=self._rank_tol,
            h=self._h,
        )
        classification = detector.classify_region(x_samples)

        # Compute derived spectral features from equilibrium Jacobian
        eigvals = classification.eigenvalues
        meta = {
            'domain': domain,
            'theta': dict(theta),
            'hurwitz_margin': hurwitz_margin(eigvals),
            'dominant_freq_hz': dominant_frequency_hz(eigvals),
            'damping_ratio': damping_ratio(eigvals),
        }

        patch = self.atlas.add_classification(classification, auto_merge=True)
        patch.metadata.update(meta)

        return MapResult(
            patch=patch,
            classification=classification,
            theta=theta,
            domain=domain,
            equilibrium=x_eq,
            converged=converged,
        )

    # ── Parameter space scanning ──────────────────────────────────────────────

    def scan_grid(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_per_axis: int,
        domain: str,
        x0: Optional[np.ndarray] = None,
    ) -> List[MapResult]:
        """
        Grid scan over parameter space.

        Args:
            param_ranges: {'R': (1.0, 1000.0), 'C': (1e-9, 1e-6)}
            n_per_axis:   points per parameter axis (total = n_per_axis^n_params)
            domain:       domain label for all generated patches
            x0:           equilibrium initial guess (shared across all points)

        Returns:
            List of MapResult, one per grid point.
        """
        keys = list(param_ranges.keys())
        axes = [
            np.linspace(param_ranges[k][0], param_ranges[k][1], n_per_axis)
            for k in keys
        ]
        results = []
        for values in itertools.product(*axes):
            theta = dict(zip(keys, values))
            try:
                results.append(self.map_point(theta, domain, x0=x0))
            except Exception:
                pass  # singular C, non-convergent equilibrium, etc.
        return results

    def scan_random(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_samples: int,
        domain: str,
        x0: Optional[np.ndarray] = None,
    ) -> List[MapResult]:
        """
        Random scan over parameter space (uniform distribution).

        Args:
            param_bounds: {'R': (1.0, 1000.0), 'C': (1e-9, 1e-6)}
            n_samples:    number of random parameter points to evaluate
            domain:       domain label for all generated patches
            x0:           equilibrium initial guess

        Returns:
            List of MapResult.
        """
        keys = list(param_bounds.keys())
        lo = np.array([param_bounds[k][0] for k in keys])
        hi = np.array([param_bounds[k][1] for k in keys])

        results = []
        for _ in range(n_samples):
            values = self._rng.uniform(lo, hi)
            theta = dict(zip(keys, values))
            try:
                results.append(self.map_point(theta, domain, x0=x0))
            except Exception:
                pass
        return results

    # ── Summary ───────────────────────────────────────────────────────────────

    def scan_report(self, results: List[MapResult]) -> dict:
        """
        Summarize a scan over parameter space.

        Returns dict with keys:
          n_points, n_converged, lca_fraction, nonabelian_fraction,
          chaotic_fraction, hurwitz_fraction (stable), freq_range_hz
        """
        if not results:
            return {'n_points': 0}

        n = len(results)
        converged = sum(1 for r in results if r.converged)
        types = [r.classification.patch_type for r in results]
        lca_n = types.count('lca')
        na_n = types.count('nonabelian')
        ch_n = types.count('chaotic')
        hurwitz_n = sum(
            1 for r in results
            if r.patch.metadata.get('hurwitz_margin', 0.0) < 0.0
        )
        freqs = [
            r.patch.metadata.get('dominant_freq_hz', 0.0)
            for r in results
            if r.patch.metadata.get('dominant_freq_hz', 0.0) > 0.0
        ]

        return {
            'n_points': n,
            'n_converged': converged,
            'lca_fraction': lca_n / n,
            'nonabelian_fraction': na_n / n,
            'chaotic_fraction': ch_n / n,
            'hurwitz_fraction': hurwitz_n / n,
            'freq_range_hz': (min(freqs), max(freqs)) if freqs else (0.0, 0.0),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _unwrap(self, theta: dict) -> Tuple[Callable, int]:
        """
        Call system_factory and extract (system_fn, n_states).

        Supports:
          - DynamicalSystem: has n_states() method and rhs(t, x)
          - plain callable: f(x) -> ẋ, requires self.n_states
        """
        system = self.system_factory(theta)

        if hasattr(system, 'n_states') and callable(system.n_states):
            # Form A: DynamicalSystem
            n = system.n_states()
            system_fn = lambda x: system.rhs(0.0, x)
        elif callable(system):
            # Form B: plain vector field
            if self.n_states is None:
                raise ValueError(
                    "system_factory returned a plain callable but n_states was not "
                    "set. Pass n_states=<int> to EigenspaceMapper.__init__."
                )
            n = self.n_states
            system_fn = system
        else:
            raise TypeError(
                f"system_factory must return a DynamicalSystem or callable, "
                f"got {type(system)}"
            )
        return system_fn, n
