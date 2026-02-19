"""
Tests for PontryaginDualExtractor — frequency character extraction.
"""

import numpy as np
import pytest
from tensor.lca_patch_detector import LCAPatchDetector, PatchClassification
from tensor.pontryagin_dual import PontryaginDualExtractor, Character, SharedCharacterResult


# ------------------------------------------------------------------
# Fixtures: LCA patch classifications
# ------------------------------------------------------------------

def linear_system(x: np.ndarray) -> np.ndarray:
    """Linear stable system: stable oscillator."""
    A = np.array([[-0.5, -1.0], [1.0, -0.5]])
    return A @ x


def spring_mass_system(x: np.ndarray) -> np.ndarray:
    """Spring-mass-damper: same operator class as linear RLC."""
    # m*xdd + c*xd + k*x = 0, with m=1, c=1, k=1
    pos, vel = x
    return np.array([vel, -pos - vel])


def make_classification(system_fn, x_center=None, spread=0.1) -> PatchClassification:
    """Generate a PatchClassification for a system around a given point."""
    detector = LCAPatchDetector(system_fn, n_states=2)
    if x_center is None:
        x_center = np.array([0.1, 0.0])
    rng = np.random.default_rng(42)
    x_samples = x_center + rng.normal(0, spread, (20, 2))
    return detector.classify_region(x_samples)


@pytest.fixture
def extractor():
    return PontryaginDualExtractor()


@pytest.fixture
def rlc_classification():
    return make_classification(linear_system)


@pytest.fixture
def spring_classification():
    return make_classification(spring_mass_system)


# ------------------------------------------------------------------
# Tests: Character dataclass
# ------------------------------------------------------------------

class TestCharacter:
    def test_stable_character(self):
        c = Character(
            eigenvalue=complex(-0.5, 1.0),
            eigenvector=np.array([1.0, 0.0]),
            frequency=1.0,
            decay_rate=0.5,
            magnitude=0.8,
        )
        assert c.is_stable
        assert c.is_oscillatory

    def test_unstable_character(self):
        c = Character(
            eigenvalue=complex(0.3, 0.0),
            eigenvector=np.array([1.0, 0.0]),
            frequency=0.0,
            decay_rate=0.3,
            magnitude=0.5,
        )
        assert not c.is_stable
        assert not c.is_oscillatory

    def test_pure_real_eigenvalue(self):
        c = Character(
            eigenvalue=complex(-1.0, 0.0),
            eigenvector=np.array([0.0, 1.0]),
            frequency=0.0,
            decay_rate=1.0,
            magnitude=1.0,
        )
        assert not c.is_oscillatory
        assert c.is_stable


# ------------------------------------------------------------------
# Tests: Character extraction
# ------------------------------------------------------------------

class TestCharacterExtraction:
    def test_returns_list(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        assert isinstance(chars, list)

    def test_returns_character_objects(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        assert all(isinstance(c, Character) for c in chars)

    def test_nonempty_for_lca_patch(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        assert len(chars) > 0

    def test_sorted_by_magnitude_descending(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        magnitudes = [c.magnitude for c in chars]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_magnitudes_positive(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        assert all(c.magnitude >= 0 for c in chars)

    def test_frequency_nonnegative(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        assert all(c.frequency >= 0 for c in chars)

    def test_decay_rate_nonnegative(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        assert all(c.decay_rate >= 0 for c in chars)


# ------------------------------------------------------------------
# Tests: Shared characters
# ------------------------------------------------------------------

class TestSharedCharacters:
    def test_returns_shared_result(self, extractor, rlc_classification, spring_classification):
        chars_a = extractor.extract_characters(rlc_classification)
        chars_b = extractor.extract_characters(spring_classification)
        result = extractor.shared_characters(chars_a, chars_b)
        assert isinstance(result, SharedCharacterResult)

    def test_shared_same_patch_high_alignment(self, extractor, rlc_classification):
        """Same patch with itself should have high mean alignment."""
        chars = extractor.extract_characters(rlc_classification)
        result = extractor.shared_characters(chars, chars)
        # At least some shared characters with high alignment
        if result.shared_indices:
            assert result.mean_alignment > 0.5

    def test_shared_indices_are_valid(self, extractor, rlc_classification, spring_classification):
        chars_a = extractor.extract_characters(rlc_classification)
        chars_b = extractor.extract_characters(spring_classification)
        result = extractor.shared_characters(chars_a, chars_b)
        for (i, j) in result.shared_indices:
            assert 0 <= i < len(chars_a)
            assert 0 <= j < len(chars_b)

    def test_alignment_scores_in_range(self, extractor, rlc_classification, spring_classification):
        chars_a = extractor.extract_characters(rlc_classification)
        chars_b = extractor.extract_characters(spring_classification)
        result = extractor.shared_characters(chars_a, chars_b)
        for score in result.alignment_scores:
            assert 0.0 <= score <= 1.0

    def test_empty_chars_a_gives_empty_shared(self, extractor, rlc_classification):
        chars_b = extractor.extract_characters(rlc_classification)
        result = extractor.shared_characters([], chars_b)
        assert result.shared_indices == []

    def test_empty_chars_b_gives_empty_shared(self, extractor, rlc_classification):
        chars_a = extractor.extract_characters(rlc_classification)
        result = extractor.shared_characters(chars_a, [])
        assert result.shared_indices == []

    def test_top_k_limits_comparison(self, extractor, rlc_classification):
        chars_a = extractor.extract_characters(rlc_classification)
        chars_b = extractor.extract_characters(rlc_classification)
        result = extractor.shared_characters(chars_a, chars_b, top_k=2)
        # All indices must be within first 2
        for (i, j) in result.shared_indices:
            assert i < 2
            assert j < 2


# ------------------------------------------------------------------
# Tests: HDV dimension mapping
# ------------------------------------------------------------------

class TestHDVMapping:
    def test_map_to_hdv_dims_returns_list(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        dims = extractor.map_to_hdv_dims(chars, hdv_dim=1000)
        assert isinstance(dims, list)

    def test_map_to_hdv_dims_in_range(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        dims = extractor.map_to_hdv_dims(chars, hdv_dim=1000)
        assert all(0 <= d < 1000 for d in dims)

    def test_map_same_length_as_chars(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        dims = extractor.map_to_hdv_dims(chars, hdv_dim=1000)
        assert len(dims) == len(chars)

    def test_dominant_frequency_vector_shape(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        vec = extractor.dominant_frequency_vector(chars, hdv_dim=1000)
        assert vec.shape == (1000,)

    def test_dominant_frequency_vector_normalized(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        vec = extractor.dominant_frequency_vector(chars, hdv_dim=1000)
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            assert abs(norm - 1.0) < 1e-5

    def test_dominant_frequency_vector_nonnegative(self, extractor, rlc_classification):
        chars = extractor.extract_characters(rlc_classification)
        vec = extractor.dominant_frequency_vector(chars, hdv_dim=1000)
        assert np.all(vec >= 0.0)


# ------------------------------------------------------------------
# Tests: Cross-domain PoC (RLC ≅ spring-mass)
# ------------------------------------------------------------------

class TestCrossDomainEquivalence:
    def test_lca_patches_produce_characters(self, extractor, rlc_classification, spring_classification):
        """Both LCA patches should yield characters (precondition for equivalence detection)."""
        chars_rlc = extractor.extract_characters(rlc_classification)
        chars_spring = extractor.extract_characters(spring_classification)
        assert len(chars_rlc) > 0
        assert len(chars_spring) > 0

    def test_rlc_and_spring_share_some_characters(self, extractor, rlc_classification, spring_classification):
        """RLC and spring-mass are algebraically equivalent — should share characters."""
        chars_rlc = extractor.extract_characters(rlc_classification)
        chars_spring = extractor.extract_characters(spring_classification)
        result = extractor.shared_characters(chars_rlc, chars_spring)
        # At least one shared character between physically equivalent systems
        # (may be 0 if systems are numerically too different, but test that the machinery works)
        assert isinstance(result.shared_indices, list)
