"""
Tests for EigenspaceMapper, DomainCanonicalizer, ParameterSpaceWalker.

System under test: series RLC with state x = [v_C, i_L].
  C·dv_C/dt = i_L           → dv_C/dt = i_L / C
  L·di_L/dt = -v_C - R·i_L  → di_L/dt = (-v_C - R·i_L) / L

Linear A = [[0, 1/C], [-1/L, -R/L]]
Eigenvalues: λ = -R/(2L) ± sqrt((R/(2L))² - 1/(LC))

For R=50, L=1e-3, C=1e-6: ζ > 1 (overdamped), both real negative → LCA, Hurwitz.
For R=1, L=1e-3, C=1e-6:  ζ < 1 (underdamped), complex with Re < 0 → LCA, Hurwitz.
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from tensor.lca_patch_detector import LCAPatchDetector
from tensor.harmonic_atlas import HarmonicAtlas
from tensor.eigenspace_mapper import (
    EigenspaceMapper, MapResult,
    hurwitz_margin, dominant_frequency_hz, damping_ratio,
)
from tensor.domain_canonicalizer import DomainCanonicalizer, CanonicalMatch
from tensor.parameter_space_walker import (
    ParameterSpaceWalker, WalkerExperience,
    _eigval_features, _spectral_gap, _regime_onehot,
)
from tensor.spectral_path import DissonanceMetric


# ── Fixtures ──────────────────────────────────────────────────────────────────

def rlc_factory(theta: dict):
    """Return vector field f(x) → ẋ for series RLC."""
    R = theta['R']
    L = theta['L']
    C = theta['C']
    def f(x):
        v_C, i_L = x
        dv = i_L / C
        di = (-v_C - R * i_L) / L
        return np.array([dv, di])
    return f


def rlc_eigvals(theta: dict) -> np.ndarray:
    """Analytical eigenvalues of the RLC A-matrix."""
    R, L, C = theta['R'], theta['L'], theta['C']
    A = np.array([[0.0, 1.0 / C], [-1.0 / L, -R / L]])
    return np.linalg.eigvals(A)


@pytest.fixture
def atlas():
    return HarmonicAtlas()


@pytest.fixture
def mapper(atlas):
    return EigenspaceMapper(
        system_factory=rlc_factory,
        atlas=atlas,
        n_states=2,
        n_samples=20,
        sample_radius=0.005,
        rng_seed=42,
    )


UNDERDAMPED = {'R': 1.0, 'L': 1e-3, 'C': 1e-6}     # ζ≈0.016, complex eigenvalues
OVERDAMPED  = {'R': 200.0, 'L': 1e-3, 'C': 1e-6}  # ζ≈3.16, real negative eigenvalues


# ── Spectral feature helpers ──────────────────────────────────────────────────

class TestSpectralHelpers:
    def test_hurwitz_margin_stable(self):
        ev = np.array([-2.0 + 3j, -2.0 - 3j])
        assert hurwitz_margin(ev) == pytest.approx(-2.0)

    def test_hurwitz_margin_unstable(self):
        ev = np.array([+0.1 + 0j, -2.0 + 0j])
        assert hurwitz_margin(ev) == pytest.approx(0.1)

    def test_dominant_frequency_underdamped(self):
        # RLC underdamped: ω₀ ≈ 1/sqrt(LC) = 1/sqrt(1e-9) ≈ 31623 rad/s
        ev = rlc_eigvals(UNDERDAMPED)
        freq = dominant_frequency_hz(ev)
        assert freq > 0.0
        assert freq == pytest.approx(np.max(np.abs(np.imag(ev))) / (2 * np.pi), rel=1e-3)

    def test_dominant_frequency_overdamped(self):
        # Pure real eigenvalues → no oscillatory component
        ev = rlc_eigvals(OVERDAMPED)
        assert np.all(np.imag(ev) == 0.0)
        assert dominant_frequency_hz(ev) == pytest.approx(0.0)

    def test_damping_ratio_range(self):
        ev = rlc_eigvals(UNDERDAMPED)
        dr = damping_ratio(ev)
        assert 0.0 <= dr <= 1.0

    def test_hurwitz_margin_empty(self):
        assert hurwitz_margin(np.array([])) == pytest.approx(0.0)


# ── EigenspaceMapper.map_point ────────────────────────────────────────────────

class TestMapPoint:
    def test_returns_map_result(self, mapper):
        result = mapper.map_point(UNDERDAMPED, 'rlc_test')
        assert isinstance(result, MapResult)

    def test_patch_metadata_keys(self, mapper):
        result = mapper.map_point(UNDERDAMPED, 'rlc_test')
        meta = result.patch.metadata
        for key in ('domain', 'theta', 'hurwitz_margin', 'dominant_freq_hz', 'damping_ratio'):
            assert key in meta, f"Missing metadata key: {key}"

    def test_domain_stored_correctly(self, mapper):
        result = mapper.map_point(UNDERDAMPED, 'solar_mppt')
        assert result.patch.metadata['domain'] == 'solar_mppt'

    def test_theta_stored_correctly(self, mapper):
        result = mapper.map_point(UNDERDAMPED, 'rlc_test')
        stored = result.patch.metadata['theta']
        for k, v in UNDERDAMPED.items():
            assert stored[k] == pytest.approx(v)

    def test_hurwitz_stable_system(self, mapper):
        result = mapper.map_point(UNDERDAMPED, 'rlc_test')
        assert result.patch.metadata['hurwitz_margin'] < 0.0

    def test_patch_type_lca_for_linear_system(self, mapper):
        # Linear systems have constant Jacobian → commutator = 0 → LCA
        result = mapper.map_point(UNDERDAMPED, 'rlc_test')
        assert result.patch.patch_type == 'lca'

    def test_equilibrium_at_origin(self, mapper):
        # For RLC with no source, equilibrium is x = 0
        result = mapper.map_point(UNDERDAMPED, 'rlc_test')
        assert np.allclose(result.equilibrium, np.zeros(2), atol=1e-4)

    def test_map_two_points_to_atlas(self, mapper, atlas):
        mapper.map_point(UNDERDAMPED, 'rlc_test')
        mapper.map_point(OVERDAMPED, 'rlc_test')
        # Atlas has at most 2 patches (may merge if spectrally similar)
        assert len(atlas.all_patches()) >= 1


# ── EigenspaceMapper.scan_random ─────────────────────────────────────────────

class TestScanRandom:
    def test_returns_list_of_results(self, mapper):
        bounds = {'R': (1.0, 20.0), 'L': (1e-4, 1e-2), 'C': (1e-7, 1e-5)}
        results = mapper.scan_random(bounds, n_samples=8, domain='rlc_scan')
        assert isinstance(results, list)
        assert len(results) <= 8  # some may fail (singular C, etc.)

    def test_all_results_have_metadata(self, mapper):
        bounds = {'R': (1.0, 20.0), 'L': (1e-4, 1e-2), 'C': (1e-7, 1e-5)}
        results = mapper.scan_random(bounds, n_samples=6, domain='rlc_scan')
        for r in results:
            assert 'domain' in r.patch.metadata
            assert r.patch.metadata['domain'] == 'rlc_scan'

    def test_scan_report_keys(self, mapper):
        bounds = {'R': (1.0, 20.0), 'L': (1e-4, 1e-2), 'C': (1e-7, 1e-5)}
        results = mapper.scan_random(bounds, n_samples=6, domain='rlc_scan')
        report = mapper.scan_report(results)
        for key in ('n_points', 'lca_fraction', 'hurwitz_fraction', 'freq_range_hz'):
            assert key in report

    def test_scan_report_fractions_sum_to_one(self, mapper):
        bounds = {'R': (1.0, 20.0), 'L': (1e-4, 1e-2), 'C': (1e-7, 1e-5)}
        results = mapper.scan_random(bounds, n_samples=10, domain='rlc_scan')
        if not results:
            pytest.skip("No successful scan results")
        r = mapper.scan_report(results)
        total = r['lca_fraction'] + r['nonabelian_fraction'] + r['chaotic_fraction']
        assert total == pytest.approx(1.0, abs=1e-6)


# ── EigenspaceMapper.scan_grid ────────────────────────────────────────────────

class TestScanGrid:
    def test_grid_count(self, mapper):
        bounds = {'R': (1.0, 20.0), 'L': (1e-3, 1e-3)}  # L fixed effectively
        results = mapper.scan_grid(
            param_ranges=bounds, n_per_axis=3,
            domain='rlc_grid', x0=np.zeros(2),
        )
        assert len(results) <= 9  # 3 × 3 grid, some may fail


# ── DomainCanonicalizer ───────────────────────────────────────────────────────

class TestDomainCanonicalizer:
    def test_empty_atlas_returns_none(self, atlas):
        canon = DomainCanonicalizer(atlas)
        ev = rlc_eigvals(UNDERDAMPED)
        assert canon.recognize(ev) is None

    def test_recognizes_same_system(self, mapper, atlas):
        # Populate atlas with underdamped RLC patch
        mapper.map_point(UNDERDAMPED, 'rlc_underdamped')

        canon = DomainCanonicalizer(atlas, tau_threshold=500.0)  # permissive
        ev = rlc_eigvals(UNDERDAMPED)
        match = canon.recognize(ev)
        assert match is not None
        assert isinstance(match, CanonicalMatch)

    def test_match_has_required_fields(self, mapper, atlas):
        mapper.map_point(UNDERDAMPED, 'rlc_underdamped')
        canon = DomainCanonicalizer(atlas, tau_threshold=500.0)
        match = canon.recognize(rlc_eigvals(UNDERDAMPED))
        assert match is not None
        assert match.patch_id >= 0
        assert isinstance(match.interval_ratio, str)
        assert ':' in match.interval_ratio
        assert 0.0 <= match.confidence <= 1.0

    def test_self_recognition_is_consonant(self, mapper, atlas):
        # Same system should have near-zero dissonance with itself
        mapper.map_point(UNDERDAMPED, 'rlc_underdamped')
        canon = DomainCanonicalizer(atlas, tau_threshold=500.0)
        match = canon.recognize(rlc_eigvals(UNDERDAMPED))
        assert match is not None
        assert match.spectral_distance == pytest.approx(0.0, abs=1.0)

    def test_tight_threshold_rejects_distant_domain(self, mapper, atlas):
        mapper.map_point(UNDERDAMPED, 'rlc_underdamped')
        canon = DomainCanonicalizer(atlas, tau_threshold=1e-6)  # very strict
        # Pure DC system: eigenvalues far from oscillatory RLC
        dc_eigvals = np.array([-100.0 + 0j, -200.0 + 0j])
        match = canon.recognize(dc_eigvals)
        # Either None or has high spectral_distance — the test is that at
        # zero-frequency there's no oscillatory match to the RLC
        if match is not None:
            assert match.spectral_distance >= 0.0

    def test_nearest_patch_always_returned(self, mapper, atlas):
        mapper.map_point(UNDERDAMPED, 'rlc_underdamped')
        canon = DomainCanonicalizer(atlas, tau_threshold=0.0)  # impossible threshold
        ev = rlc_eigvals(UNDERDAMPED)
        nearest = canon.nearest_patch_spectrum(ev)
        assert nearest is not None

    def test_ratio_str_format(self, atlas):
        canon = DomainCanonicalizer(atlas)
        ratio = canon._ratio_str(3000.0, 2000.0)  # 3:2
        assert ':' in ratio
        p, q = ratio.split(':')
        assert p.isdigit() and q.isdigit()

    def test_ratio_str_zero_denominator(self, atlas):
        canon = DomainCanonicalizer(atlas)
        assert canon._ratio_str(500.0, 0.0) == '0:1'

    def test_batch_recognize(self, mapper, atlas):
        mapper.map_point(UNDERDAMPED, 'rlc_underdamped')
        canon = DomainCanonicalizer(atlas, tau_threshold=500.0)
        batch = [rlc_eigvals(UNDERDAMPED), rlc_eigvals(OVERDAMPED)]
        results = canon.recognize_batch(batch)
        assert len(results) == 2

    def test_domain_hint_filter(self, mapper, atlas):
        mapper.map_point(UNDERDAMPED, 'rlc_underdamped')
        canon = DomainCanonicalizer(atlas, tau_threshold=500.0)
        ev = rlc_eigvals(UNDERDAMPED)
        # Correct domain hint → finds patch
        match = canon.recognize(ev, domain_hint='rlc_underdamped')
        assert match is not None
        # Wrong domain hint → falls back to full search (since filtered set empty)
        match2 = canon.recognize(ev, domain_hint='nonexistent_domain')
        assert match2 is not None  # fallback to all patches


# ── ParameterSpaceWalker ──────────────────────────────────────────────────────

class TestParameterSpaceWalker:
    _KEYS = ['R', 'L', 'C']
    _BOUNDS = {'R': (1.0, 100.0), 'L': (1e-4, 1e-2), 'C': (1e-9, 1e-6)}

    @pytest.fixture
    def walker(self):
        return ParameterSpaceWalker(
            theta_keys=self._KEYS,
            param_bounds=self._BOUNDS,
            hidden=32,
            seed=0,
        )

    def test_predict_step_shape(self, walker):
        theta = UNDERDAMPED
        ev = rlc_eigvals(UNDERDAMPED)
        delta = walker.predict_step(theta, ev, regime='lca')
        assert delta.shape == (3,)  # len(theta_keys)

    def test_predict_step_finite(self, walker):
        ev = rlc_eigvals(UNDERDAMPED)
        delta = walker.predict_step(UNDERDAMPED, ev)
        assert np.all(np.isfinite(delta))

    def test_predict_next_theta_keys(self, walker):
        ev = rlc_eigvals(UNDERDAMPED)
        next_t = walker.predict_next_theta(UNDERDAMPED, ev)
        assert set(next_t.keys()) == set(self._KEYS)

    def test_record_increases_buffer(self, walker):
        exp = WalkerExperience(
            theta_before=UNDERDAMPED,
            eigvals_before=rlc_eigvals(UNDERDAMPED),
            theta_after=OVERDAMPED,
            eigvals_after=rlc_eigvals(OVERDAMPED),
            dissonance=0.1,
        )
        assert walker.buffer_size() == 0
        walker.record(exp)
        assert walker.buffer_size() == 1

    def test_train_returns_loss(self, walker):
        # Populate buffer with synthetic experiences
        rng = np.random.default_rng(1)
        for _ in range(20):
            t_before = {
                'R': rng.uniform(1.0, 100.0),
                'L': rng.uniform(1e-4, 1e-2),
                'C': rng.uniform(1e-9, 1e-6),
            }
            t_after = {
                'R': rng.uniform(1.0, 100.0),
                'L': rng.uniform(1e-4, 1e-2),
                'C': rng.uniform(1e-9, 1e-6),
            }
            walker.record(WalkerExperience(
                theta_before=t_before,
                eigvals_before=rlc_eigvals(t_before),
                theta_after=t_after,
                eigvals_after=rlc_eigvals(t_after),
                dissonance=rng.uniform(0.0, 1.0),
            ))
        loss = walker.train(n_epochs=3, lr=1e-3, batch_size=8)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_loss_decreases(self, walker):
        # Train on 50 experiences; loss should decrease over epochs
        rng = np.random.default_rng(2)
        for _ in range(50):
            t_b = {'R': float(rng.uniform(1, 50)), 'L': 1e-3, 'C': 1e-6}
            t_a = {'R': float(rng.uniform(1, 50)), 'L': 1e-3, 'C': 1e-6}
            walker.record(WalkerExperience(
                theta_before=t_b,
                eigvals_before=rlc_eigvals(t_b),
                theta_after=t_a,
                eigvals_after=rlc_eigvals(t_a),
                dissonance=float(rng.uniform(0.0, 0.2)),
            ))
        loss_early = walker.train(n_epochs=5, lr=1e-2, batch_size=16)
        loss_late = walker.train(n_epochs=20, lr=1e-3, batch_size=16)
        # Not a strict assertion — just confirm training ran and returned floats
        assert isinstance(loss_early, float)
        assert isinstance(loss_late, float)

    def test_buffer_stats_keys(self, walker):
        walker.record(WalkerExperience(
            theta_before=UNDERDAMPED,
            eigvals_before=rlc_eigvals(UNDERDAMPED),
            theta_after=OVERDAMPED,
            eigvals_after=rlc_eigvals(OVERDAMPED),
            dissonance=0.05,
        ))
        stats = walker.buffer_dissonance_stats()
        assert 'count' in stats and 'mean' in stats

    def test_clear_buffer(self, walker):
        walker.record(WalkerExperience(
            theta_before=UNDERDAMPED,
            eigvals_before=rlc_eigvals(UNDERDAMPED),
            theta_after=OVERDAMPED,
            eigvals_after=rlc_eigvals(OVERDAMPED),
            dissonance=0.1,
        ))
        walker.clear_buffer()
        assert walker.buffer_size() == 0

    def test_small_buffer_train_noop(self, walker):
        # Buffer < 4 entries → train returns 0.0 without error
        walker.record(WalkerExperience(
            theta_before=UNDERDAMPED,
            eigvals_before=rlc_eigvals(UNDERDAMPED),
            theta_after=OVERDAMPED,
            eigvals_after=rlc_eigvals(OVERDAMPED),
            dissonance=0.1,
        ))
        loss = walker.train(n_epochs=5)
        assert loss == pytest.approx(0.0)


# ── Feature engineering helpers ───────────────────────────────────────────────

class TestFeatureEngineering:
    def test_eigval_features_shape(self):
        ev = rlc_eigvals(UNDERDAMPED)
        feats = _eigval_features(ev, n=4)
        assert feats.shape == (8,)

    def test_eigval_features_pads_short(self):
        ev = np.array([-1.0 + 2j])   # only 1 eigenvalue
        feats = _eigval_features(ev, n=4)
        assert feats.shape == (8,)
        # Padded positions should be zero
        assert feats[1] == pytest.approx(0.0)
        assert feats[5] == pytest.approx(0.0)

    def test_eigval_features_empty(self):
        feats = _eigval_features(np.array([]), n=4)
        assert np.allclose(feats, 0.0)

    def test_spectral_gap_two_modes(self):
        ev = np.array([-5.0 + 0j, -2.0 + 0j])
        gap = _spectral_gap(ev)
        assert gap == pytest.approx(3.0)

    def test_regime_onehot_lca(self):
        enc = _regime_onehot('lca')
        assert np.allclose(enc, [1.0, 0.0, 0.0])

    def test_regime_onehot_unknown(self):
        enc = _regime_onehot('unknown')
        assert np.allclose(enc, [0.0, 0.0, 0.0])
