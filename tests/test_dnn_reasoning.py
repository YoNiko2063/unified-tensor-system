"""Tests for tensor/dnn_reasoning.py"""

import pytest
import numpy as np

from tensor.dnn_reasoning import DeepNeuralNetworkReasoner, _softmax


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_hdv(tmp_path):
    from tensor.integrated_hdv import IntegratedHDVSystem
    return IntegratedHDVSystem(
        hdv_dim=300, n_modes=5, embed_dim=32,
        library_path=str(tmp_path / "lib.json"),
    )

@pytest.fixture
def reasoner(small_hdv):
    r = DeepNeuralNetworkReasoner(
        hdv_system=small_hdv, max_steps=5, energy_threshold=0.05, top_k=3,
    )
    # Seed knowledge base
    r.store("gradient descent minimises loss", "math")
    r.store("exponential decay dx/dt = -x", "math")
    r.store("RC circuit voltage discharge", "physical")
    r.store("pipeline fetch decode execute writeback", "physical")
    r.store("Lyapunov energy stability convergence", "math")
    return r

@pytest.fixture
def empty_reasoner(small_hdv):
    return DeepNeuralNetworkReasoner(hdv_system=small_hdv, max_steps=5)


# ── _softmax ──────────────────────────────────────────────────────────────────

def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    s = _softmax(x)
    assert abs(s.sum() - 1.0) < 1e-6

def test_softmax_nonnegative():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    s = _softmax(x)
    assert np.all(s >= 0.0)

def test_softmax_max_gets_highest_weight():
    x = np.array([0.0, 0.0, 10.0])
    s = _softmax(x)
    assert s[2] > s[0] and s[2] > s[1]

def test_softmax_uniform_input():
    x = np.zeros(5)
    s = _softmax(x)
    assert np.allclose(s, 0.2)


# ── DeepNeuralNetworkReasoner — basic ─────────────────────────────────────────

def test_reason_returns_dict(reasoner):
    result = reasoner.reason_about("optimization problem")
    assert isinstance(result, dict)

def test_reason_keys_present(reasoner):
    result = reasoner.reason_about("optimization")
    for key in ("chain", "converged", "final_energy", "steps", "energy_history"):
        assert key in result

def test_chain_is_list(reasoner):
    result = reasoner.reason_about("gradient")
    assert isinstance(result["chain"], list)

def test_chain_nonempty(reasoner):
    result = reasoner.reason_about("gradient descent")
    assert len(result["chain"]) >= 1  # at least the query itself

def test_chain_first_entry_is_query(reasoner):
    result = reasoner.reason_about("gradient descent")
    assert result["chain"][0]["text"] == "gradient descent"

def test_steps_matches_chain_length(reasoner):
    result = reasoner.reason_about("gradient descent")
    assert result["steps"] == len(result["chain"]) - 1

def test_converged_is_bool(reasoner):
    result = reasoner.reason_about("optimization")
    assert isinstance(result["converged"], bool)

def test_final_energy_nonneg(reasoner):
    result = reasoner.reason_about("optimization")
    assert result["final_energy"] >= 0.0

def test_energy_history_len(reasoner):
    result = reasoner.reason_about("decay")
    assert len(result["energy_history"]) >= 1

def test_max_steps_not_exceeded(reasoner):
    result = reasoner.reason_about("stability")
    assert result["steps"] <= reasoner.max_steps


# ── store / knowledge_size ────────────────────────────────────────────────────

def test_knowledge_size(reasoner):
    assert reasoner.knowledge_size == 5

def test_store_returns_array(reasoner, small_hdv):
    vec = reasoner.store("new concept")
    assert isinstance(vec, np.ndarray)

def test_store_increases_knowledge(reasoner):
    before = reasoner.knowledge_size
    reasoner.store("another concept")
    assert reasoner.knowledge_size == before + 1

def test_store_batch(small_hdv):
    r = DeepNeuralNetworkReasoner(hdv_system=small_hdv)
    items = [{"text": "a", "domain": "math"}, {"text": "b", "domain": "physical"}]
    vecs = r.store_batch(items)
    assert len(vecs) == 2
    assert r.knowledge_size == 2


# ── empty knowledge base ──────────────────────────────────────────────────────

def test_reason_empty_knowledge_no_crash(empty_reasoner):
    result = empty_reasoner.reason_about("something")
    assert isinstance(result, dict)
    assert result["steps"] == 0

def test_reason_no_hdv():
    r = DeepNeuralNetworkReasoner(hdv_system=None)
    result = r.reason_about("anything")
    assert result["chain"] == []
    assert not result["converged"]


# ── reason_batch ──────────────────────────────────────────────────────────────

def test_reason_batch_returns_list(reasoner):
    results = reasoner.reason_batch(["gradient", "decay", "pipeline"])
    assert isinstance(results, list)
    assert len(results) == 3

def test_reason_batch_each_is_dict(reasoner):
    results = reasoner.reason_batch(["a", "b"])
    for r in results:
        assert "chain" in r


# ── compute_reasoning_similarity ─────────────────────────────────────────────

def test_similarity_same_query(reasoner):
    # Same query should have high self-similarity
    sim = reasoner.compute_reasoning_similarity("gradient", "gradient")
    assert isinstance(sim, float)

def test_similarity_returns_float(reasoner):
    sim = reasoner.compute_reasoning_similarity("decay", "pipeline")
    assert isinstance(sim, float)

def test_similarity_empty_knowledge(empty_reasoner):
    sim = empty_reasoner.compute_reasoning_similarity("a", "b")
    assert sim == 0.0


# ── energy decreases toward convergence ──────────────────────────────────────

def test_energy_history_nonneg(reasoner):
    result = reasoner.reason_about("Lyapunov stability")
    for e in result["energy_history"]:
        assert e >= 0.0

def test_energy_bounded(reasoner):
    result = reasoner.reason_about("gradient descent minimises loss")
    for e in result["energy_history"]:
        assert e <= 2.0  # max possible angular distance is 2 (1 - (-1))
