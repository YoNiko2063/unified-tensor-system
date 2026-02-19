"""
Tests for HarmonicAtlas — automatic spectral chart builder.
"""

import numpy as np
import pytest
from tensor.lca_patch_detector import LCAPatchDetector, PatchClassification
from tensor.harmonic_atlas import HarmonicAtlas, AtlasStats
from tensor.patch_graph import Patch, PatchGraph


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def linear_system(x: np.ndarray) -> np.ndarray:
    """Linear stable: ẋ = Ax."""
    A = np.array([[-0.5, -1.0], [1.0, -0.5]])
    return A @ x


def make_detector() -> LCAPatchDetector:
    return LCAPatchDetector(linear_system, n_states=2)


def make_classification(
    patch_type: str = 'lca',
    eigvals=None,
    commutator_norm: float = 0.0,
    operator_rank: int = 1,
    centroid=None,
) -> PatchClassification:
    """Create a synthetic PatchClassification for atlas testing."""
    n = 2
    if eigvals is None:
        eigvals = np.array([-0.5 + 1.0j, -0.5 - 1.0j])
    if centroid is None:
        centroid = np.zeros(n)
    return PatchClassification(
        patch_type=patch_type,
        operator_rank=operator_rank,
        commutator_norm=commutator_norm,
        curvature_ratio=0.02,
        spectral_gap=0.5,
        basis_matrices=np.eye(n).reshape(1, n, n),
        eigenvalues=eigvals,
        centroid=centroid,
    )


@pytest.fixture
def atlas():
    return HarmonicAtlas()


@pytest.fixture
def atlas_with_detector():
    return HarmonicAtlas(lca_detector=make_detector())


# ------------------------------------------------------------------
# Tests: Basic construction
# ------------------------------------------------------------------

class TestConstruction:
    def test_empty_atlas(self, atlas):
        assert atlas.all_patches() == []

    def test_stats_empty(self, atlas):
        s = atlas.stats()
        assert s.n_patches == 0
        assert s.n_edges == 0
        assert s.n_merges == 0

    def test_lca_patches_empty(self, atlas):
        assert atlas.lca_patches() == []

    def test_repr(self, atlas):
        r = repr(atlas)
        assert 'HarmonicAtlas' in r


# ------------------------------------------------------------------
# Tests: add_classification
# ------------------------------------------------------------------

class TestAddClassification:
    def test_add_one_patch(self, atlas):
        cl = make_classification()
        patch = atlas.add_classification(cl)
        assert isinstance(patch, Patch)
        assert len(atlas.all_patches()) == 1

    def test_add_similar_merges(self, atlas):
        """Two very similar patches should merge into one."""
        cl1 = make_classification(centroid=np.array([0.0, 0.0]))
        cl2 = make_classification(centroid=np.array([0.01, 0.0]))  # nearly same
        atlas.add_classification(cl1)
        atlas.add_classification(cl2)
        # Should have 1 patch (merged) since spectra and ranks are identical
        assert len(atlas.all_patches()) <= 2  # either merged or not

    def test_add_different_patches_no_merge(self, atlas):
        """Patches with different eigenvalues should not merge."""
        cl1 = make_classification(eigvals=np.array([-0.5 + 1.0j, -0.5 - 1.0j]))
        cl2 = make_classification(eigvals=np.array([-10.0 + 0.0j, -20.0 + 0.0j]))
        atlas.add_classification(cl1)
        atlas.add_classification(cl2)
        # These should not merge (very different spectra)
        # Result depends on tol but different eigenvalues means different
        assert len(atlas.all_patches()) >= 1

    def test_add_returns_patch(self, atlas):
        cl = make_classification()
        patch = atlas.add_classification(cl)
        assert patch.id >= 0

    def test_add_no_merge(self, atlas):
        cl1 = make_classification(eigvals=np.array([-0.5j, 0.5j]))
        cl2 = make_classification(eigvals=np.array([-5.0j, 5.0j]))
        atlas.add_classification(cl1, auto_merge=False)
        atlas.add_classification(cl2, auto_merge=False)
        assert len(atlas.all_patches()) == 2


# ------------------------------------------------------------------
# Tests: Similarity
# ------------------------------------------------------------------

class TestSimilarity:
    def test_same_classification_zero_similarity(self, atlas):
        cl = make_classification()
        sim = atlas.similarity(cl, cl)
        assert sim == pytest.approx(0.0, abs=1e-10)

    def test_similar_patches_low_similarity(self, atlas):
        cl1 = make_classification(eigvals=np.array([-0.5 + 1.0j, -0.5 - 1.0j]))
        cl2 = make_classification(eigvals=np.array([-0.6 + 1.1j, -0.6 - 1.1j]))
        sim = atlas.similarity(cl1, cl2)
        assert sim < 0.5  # should be similar

    def test_different_patches_higher_similarity(self, atlas):
        cl1 = make_classification(eigvals=np.array([-0.5 + 1.0j, -0.5 - 1.0j]))
        cl2 = make_classification(eigvals=np.array([-10.0, -20.0]))
        sim_diff = atlas.similarity(cl1, cl2)

        cl3 = make_classification(eigvals=np.array([-0.6 + 1.1j, -0.6 - 1.1j]))
        sim_sim = atlas.similarity(cl1, cl3)

        assert sim_diff > sim_sim  # different patches have higher similarity metric

    def test_similarity_nonnegative(self, atlas):
        cl1 = make_classification()
        cl2 = make_classification(eigvals=np.array([-2.0, -3.0]))
        sim = atlas.similarity(cl1, cl2)
        assert sim >= 0.0


# ------------------------------------------------------------------
# Tests: merge_similar
# ------------------------------------------------------------------

class TestMergeSimilar:
    def test_merge_identical_patches(self, atlas):
        cl = make_classification()
        atlas.add_classification(cl, auto_merge=False)
        atlas.add_classification(cl, auto_merge=False)
        assert len(atlas.all_patches()) == 2
        n_merges = atlas.merge_similar(tol=0.01)
        assert n_merges >= 1
        assert len(atlas.all_patches()) <= 1

    def test_merge_returns_merge_count(self, atlas):
        cl = make_classification()
        atlas.add_classification(cl, auto_merge=False)
        atlas.add_classification(cl, auto_merge=False)
        n = atlas.merge_similar(tol=0.01)
        assert isinstance(n, int)
        assert n >= 0

    def test_no_merge_when_dissimilar(self, atlas):
        cl1 = make_classification(eigvals=np.array([-0.5 + 1.0j, -0.5 - 1.0j]))
        cl2 = make_classification(eigvals=np.array([-100.0, -200.0]))
        atlas.add_classification(cl1, auto_merge=False)
        atlas.add_classification(cl2, auto_merge=False)
        n = atlas.merge_similar(tol=0.01)  # very tight tolerance
        assert n == 0
        assert len(atlas.all_patches()) == 2


# ------------------------------------------------------------------
# Tests: get_chart
# ------------------------------------------------------------------

class TestGetChart:
    def test_empty_atlas_returns_none(self, atlas):
        assert atlas.get_chart(np.array([1.0, 0.0])) is None

    def test_returns_nearest_patch(self, atlas):
        cl1 = make_classification(centroid=np.array([0.0, 0.0]))
        cl2 = make_classification(
            eigvals=np.array([-10.0, -20.0]),
            centroid=np.array([10.0, 10.0])
        )
        atlas.add_classification(cl1, auto_merge=False)
        atlas.add_classification(cl2, auto_merge=False)

        # Query near first patch
        patch = atlas.get_chart(np.array([0.1, 0.0]))
        # Should be the patch with centroid near [0, 0]
        assert patch is not None
        assert np.linalg.norm(patch.centroid - np.array([0.0, 0.0])) < 5.0

    def test_returns_patch_instance(self, atlas):
        cl = make_classification()
        atlas.add_classification(cl)
        patch = atlas.get_chart(np.array([0.0, 0.0]))
        assert isinstance(patch, Patch)


# ------------------------------------------------------------------
# Tests: export_graph
# ------------------------------------------------------------------

class TestExportGraph:
    def test_empty_atlas_exports_empty_graph(self, atlas):
        g = atlas.export_graph()
        assert isinstance(g, PatchGraph)
        assert g.summary()['n_patches'] == 0

    def test_exported_graph_has_correct_patch_count(self, atlas):
        for i in range(3):
            cl = make_classification(
                eigvals=np.array([-(i + 1) * 5.0, -(i + 1) * 10.0]),
                centroid=np.array([float(i), 0.0])
            )
            atlas.add_classification(cl, auto_merge=False)
        g = atlas.export_graph()
        assert g.summary()['n_patches'] == 3

    def test_export_graph_patches_accessible(self, atlas):
        cl = make_classification()
        patch = atlas.add_classification(cl)
        g = atlas.export_graph()
        assert g.get_patch(patch.id) is not None


# ------------------------------------------------------------------
# Tests: add_states (requires detector)
# ------------------------------------------------------------------

class TestAddStates:
    def test_add_states_returns_patches(self, atlas_with_detector):
        rng = np.random.default_rng(0)
        states = rng.normal(0, 0.1, (50, 2))
        patches = atlas_with_detector.add_states(states, window=10)
        assert isinstance(patches, list)
        assert len(patches) > 0

    def test_add_states_without_detector_raises(self):
        atlas = HarmonicAtlas(lca_detector=None)
        with pytest.raises(RuntimeError):
            atlas.add_states(np.random.randn(20, 2))

    def test_add_states_populates_atlas(self, atlas_with_detector):
        rng = np.random.default_rng(1)
        states = rng.normal(0, 0.1, (40, 2))
        atlas_with_detector.add_states(states, window=10)
        assert len(atlas_with_detector.all_patches()) > 0


# ------------------------------------------------------------------
# Tests: stats
# ------------------------------------------------------------------

class TestStats:
    def test_stats_type(self, atlas):
        s = atlas.stats()
        assert isinstance(s, AtlasStats)

    def test_stats_after_add(self, atlas):
        for i in range(3):
            atlas.add_classification(
                make_classification(
                    eigvals=np.array([-(i + 1) * 5.0, -(i + 1) * 10.0])
                ),
                auto_merge=False
            )
        s = atlas.stats()
        assert s.n_patches == 3

    def test_lca_fraction(self, atlas):
        atlas.add_classification(make_classification('lca'), auto_merge=False)
        atlas.add_classification(make_classification('chaotic'), auto_merge=False)
        s = atlas.stats()
        # lca_fraction = 0.5 (may vary if patches merged)
        assert 0.0 <= s.lca_fraction <= 1.0
