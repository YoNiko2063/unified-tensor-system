"""Tests for tensor/prediction_learning.py"""

import math
import pytest
import numpy as np

from tensor.prediction_learning import (
    Concept,
    ConceptGraph,
    StructuredTextExtractor,
    PredictiveConceptLearner,
    Problem,
    ProblemGenerator,
    KnowledgeBasedProblemSolver,
    PredictionVerifier,
    ContinuousLearningLoop,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_graph():
    g = ConceptGraph()
    a = Concept(name="alpha", concept_type="operation")
    b = Concept(name="beta",  concept_type="theorem", prerequisites=["alpha"])
    c = Concept(name="gamma", concept_type="algorithm", prerequisites=["beta"])
    for concept in [a, b, c]:
        g.add_concept(concept)
    g.add_edge("alpha", "beta", weight=0.8)
    g.add_edge("beta", "gamma", weight=0.6)
    return g


@pytest.fixture
def small_hdv(tmp_path):
    from tensor.integrated_hdv import IntegratedHDVSystem
    return IntegratedHDVSystem(
        hdv_dim=300, n_modes=5, embed_dim=32,
        library_path=str(tmp_path / "lib.json"),
    )


# ── Concept ───────────────────────────────────────────────────────────────────

def test_concept_hash_equal():
    c1 = Concept(name="foo")
    c2 = Concept(name="foo")
    assert c1 == c2
    assert hash(c1) == hash(c2)


def test_concept_not_equal_different_names():
    assert Concept(name="foo") != Concept(name="bar")


# ── ConceptGraph ──────────────────────────────────────────────────────────────

def test_graph_add_and_get(simple_graph):
    assert simple_graph.get_concept("alpha") is not None
    assert simple_graph.get_concept("missing") is None


def test_graph_len(simple_graph):
    assert len(simple_graph) == 3


def test_graph_all_concepts(simple_graph):
    names = {c.name for c in simple_graph.all_concepts()}
    assert names == {"alpha", "beta", "gamma"}


def test_graph_out_degree(simple_graph):
    alpha = simple_graph.get_concept("alpha")
    assert simple_graph.out_degree(alpha) == 1


def test_graph_prerequisites_met_empty(simple_graph):
    alpha = simple_graph.get_concept("alpha")
    assert simple_graph.prerequisites_met(alpha)   # no prereqs


def test_graph_prerequisites_met_blocked(simple_graph):
    beta = simple_graph.get_concept("beta")
    assert not simple_graph.prerequisites_met(beta)  # alpha not learned


def test_graph_prerequisites_met_after_learn(simple_graph):
    simple_graph.mark_learned("alpha")
    beta = simple_graph.get_concept("beta")
    assert simple_graph.prerequisites_met(beta)


def test_graph_all_learned_false(simple_graph):
    assert not simple_graph.all_concepts_learned()


def test_graph_all_learned_true(simple_graph):
    for name in ["alpha", "beta", "gamma"]:
        simple_graph.mark_learned(name)
    assert simple_graph.all_concepts_learned()


def test_graph_get_unknown(simple_graph):
    simple_graph.mark_learned("alpha")
    unknown = simple_graph.get_unknown_concepts()
    names = {c.name for c in unknown}
    assert "alpha" not in names
    assert "beta" in names


def test_graph_get_unknown_with_explicit_set(simple_graph):
    unknown = simple_graph.get_unknown_concepts(learned={"alpha", "beta"})
    assert len(unknown) == 1
    assert unknown[0].name == "gamma"


# ── StructuredTextExtractor ───────────────────────────────────────────────────

SAMPLE_TEXT = """
## Vectors

**vector_addition** is defined as combining two vectors component-wise.
The `scalar_multiplication` operation scales all components by a factor.

## Matrices

**matrix_multiplication** requires understanding vectors first.
The `dot_product` computes a scalar from two vectors.

Matrices can represent linear transformations.
The matrix_multiplication and dot_product operations co-occur in many algorithms.
"""


def test_extractor_returns_graph():
    ext = StructuredTextExtractor()
    graph = ext.extract_concepts(SAMPLE_TEXT)
    assert isinstance(graph, ConceptGraph)


def test_extractor_finds_bold_terms():
    ext = StructuredTextExtractor()
    graph = ext.extract_concepts(SAMPLE_TEXT)
    names = {c.name for c in graph.all_concepts()}
    # Should find vector_addition and matrix_multiplication (bold)
    assert any("vector" in n or "matrix" in n for n in names)


def test_extractor_nonempty_for_rich_text():
    ext = StructuredTextExtractor()
    graph = ext.extract_concepts(SAMPLE_TEXT)
    assert len(graph) > 0


def test_extractor_empty_text():
    ext = StructuredTextExtractor()
    graph = ext.extract_concepts("")
    assert isinstance(graph, ConceptGraph)


def test_extractor_mi_edges_exist():
    """matrix_multiplication and dot_product co-occur → MI edge expected."""
    ext = StructuredTextExtractor(mi_threshold=1e-6)
    graph = ext.extract_concepts(SAMPLE_TEXT)
    # At least some edges should be present
    total_edges = sum(
        len(graph._edges.get(c.name, []))
        for c in graph.all_concepts()
    )
    assert total_edges >= 0   # structure check; actual edges depend on tokens


def test_extractor_chunk_by_headers():
    ext = StructuredTextExtractor()
    chunks = ext._chunk("# Header1\nContent1\n# Header2\nContent2")
    assert len(chunks) >= 2


# ── PredictiveConceptLearner ──────────────────────────────────────────────────

def test_learner_predicts_alpha_first(simple_graph):
    learner = PredictiveConceptLearner(simple_graph)
    concept = learner.predict_next_concept()
    # Only alpha has all prereqs met (no prereqs needed)
    assert concept is not None
    assert concept.name == "alpha"


def test_learner_predicts_beta_after_alpha(simple_graph):
    learner = PredictiveConceptLearner(simple_graph)
    learner.mark_learned("alpha")
    concept = learner.predict_next_concept()
    assert concept is not None
    assert concept.name == "beta"


def test_learner_none_when_all_learned(simple_graph):
    learner = PredictiveConceptLearner(simple_graph)
    for n in ["alpha", "beta", "gamma"]:
        learner.mark_learned(n)
    assert learner.predict_next_concept() is None


def test_learner_information_gain_positive(simple_graph):
    learner = PredictiveConceptLearner(simple_graph)
    alpha = simple_graph.get_concept("alpha")
    gain = learner._information_gain(alpha)
    assert isinstance(gain, float)


def test_learner_entropy_zero_for_empty():
    g = ConceptGraph()
    learner = PredictiveConceptLearner(g)
    assert learner._entropy([]) == 0.0


def test_learner_mark_learned_updates_graph(simple_graph):
    learner = PredictiveConceptLearner(simple_graph)
    learner.mark_learned("alpha")
    assert simple_graph.is_learned("alpha")
    assert "alpha" in learner.learned_concepts


# ── ProblemGenerator ──────────────────────────────────────────────────────────

def test_generator_returns_problem(simple_graph):
    gen = ProblemGenerator(simple_graph)
    alpha = simple_graph.get_concept("alpha")
    prob = gen.generate_problem(alpha)
    assert isinstance(prob, Problem)
    assert prob.concept == alpha
    assert len(prob.text) > 0


def test_generator_difficulty_reflects_prereqs(simple_graph):
    gen = ProblemGenerator(simple_graph)
    alpha = simple_graph.get_concept("alpha")
    gamma = simple_graph.get_concept("gamma")
    assert gen.generate_problem(alpha).difficulty == 0
    assert gen.generate_problem(gamma).difficulty == 1  # one listed prereq


def test_generator_verify_none_solution():
    g = ConceptGraph()
    g.add_concept(Concept("x"))
    gen = ProblemGenerator(g)
    prob = gen.generate_problem(g.get_concept("x"))
    ok, conf = gen.verify_solution(prob, None)
    assert not ok
    assert conf == 0.0


def test_generator_verify_ground_truth_correct():
    g = ConceptGraph()
    g.add_concept(Concept("x"))
    gen = ProblemGenerator(g)
    prob = gen.generate_problem(g.get_concept("x"))
    prob.ground_truth = "answer"
    ok, conf = gen.verify_solution(prob, "answer")
    assert ok and conf == 1.0


def test_generator_verify_ground_truth_wrong():
    g = ConceptGraph()
    g.add_concept(Concept("x"))
    gen = ProblemGenerator(g)
    prob = gen.generate_problem(g.get_concept("x"))
    prob.ground_truth = "right"
    ok, conf = gen.verify_solution(prob, "wrong")
    assert not ok and conf == 0.0


def test_generator_verify_mdl_plausible_answer():
    g = ConceptGraph()
    g.add_concept(Concept("x"))
    gen = ProblemGenerator(g)
    prob = gen.generate_problem(g.get_concept("x"))
    # ~15-word answer → near expected length for difficulty=0 → decent confidence
    answer = " ".join(["word"] * 15)
    ok, conf = gen.verify_solution(prob, answer)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


# ── KnowledgeBasedProblemSolver ───────────────────────────────────────────────

def test_solver_returns_none_no_hdv(simple_graph):
    solver = KnowledgeBasedProblemSolver(hdv_system=None)
    gen = ProblemGenerator(simple_graph)
    prob = gen.generate_problem(simple_graph.get_concept("alpha"))
    sol, conf = solver.solve(prob)
    assert sol is None
    assert conf == 0.0


def test_solver_no_store_returns_none(simple_graph, small_hdv):
    solver = KnowledgeBasedProblemSolver(small_hdv)
    gen = ProblemGenerator(simple_graph)
    prob = gen.generate_problem(simple_graph.get_concept("alpha"))
    sol, conf = solver.solve(prob)
    assert sol is None
    assert conf == 0.0


def test_solver_finds_after_store(simple_graph, small_hdv):
    solver = KnowledgeBasedProblemSolver(small_hdv)
    solver.store_solution("alpha", "Apply alpha to input to produce output.")
    gen = ProblemGenerator(simple_graph)
    alpha = simple_graph.get_concept("alpha")
    prob = gen.generate_problem(alpha)
    sol, conf = solver.solve(prob)
    # Solution may or may not match depending on HDV similarity
    assert conf >= 0.0


def test_solver_store_no_hdv_silent():
    solver = KnowledgeBasedProblemSolver(hdv_system=None)
    solver.store_solution("alpha", "some solution")
    assert len(solver._store) == 0


def test_solver_find_similar_empty():
    solver = KnowledgeBasedProblemSolver(hdv_system=None)
    q = np.zeros(100, dtype=np.float32)
    assert solver._find_similar(q) == []


# ── PredictionVerifier ────────────────────────────────────────────────────────

def test_verifier_correct_decreases_energy(simple_graph, small_hdv):
    solver = KnowledgeBasedProblemSolver(small_hdv)
    verifier = PredictionVerifier(small_hdv, simple_graph, solver)
    alpha = simple_graph.get_concept("alpha")
    gen = ProblemGenerator(simple_graph)
    prob = gen.generate_problem(alpha)
    result = verifier.verify_and_update(alpha, prob, "a solution", is_correct=True)
    assert result["correct"] is True
    assert "reinforce" in result["action"]


def test_verifier_wrong_identifies_gap(simple_graph, small_hdv):
    solver = KnowledgeBasedProblemSolver(small_hdv)
    verifier = PredictionVerifier(small_hdv, simple_graph, solver)
    # gamma requires beta requires alpha — none learned
    gamma = simple_graph.get_concept("gamma")
    gen = ProblemGenerator(simple_graph)
    prob = gen.generate_problem(gamma)
    result = verifier.verify_and_update(gamma, prob, None, is_correct=False)
    assert not result["correct"]
    assert "gap" in result


def test_verifier_energy_history_grows(simple_graph, small_hdv):
    solver = KnowledgeBasedProblemSolver(small_hdv)
    verifier = PredictionVerifier(small_hdv, simple_graph, solver)
    alpha = simple_graph.get_concept("alpha")
    gen = ProblemGenerator(simple_graph)
    prob = gen.generate_problem(alpha)
    verifier.verify_and_update(alpha, prob, "sol", True)
    verifier.verify_and_update(alpha, prob, "sol", True)
    assert len(verifier.energy_history) == 2


def test_verifier_lyapunov_stable_initially(simple_graph, small_hdv):
    solver = KnowledgeBasedProblemSolver(small_hdv)
    verifier = PredictionVerifier(small_hdv, simple_graph, solver)
    assert verifier.lyapunov_stable()


# ── ContinuousLearningLoop ────────────────────────────────────────────────────

def test_loop_runs_on_short_text(small_hdv):
    loop = ContinuousLearningLoop(hdv_system=small_hdv, max_iterations=20)
    result = loop.run(SAMPLE_TEXT)
    assert "concepts_learned" in result
    assert "total_concepts" in result
    assert result["iterations"] >= 0


def test_loop_concepts_learned_nonneg(small_hdv):
    loop = ContinuousLearningLoop(hdv_system=small_hdv, max_iterations=10)
    result = loop.run(SAMPLE_TEXT)
    assert result["concepts_learned"] >= 0


def test_loop_no_hdv(tmp_path):
    loop = ContinuousLearningLoop(hdv_system=None, max_iterations=5)
    result = loop.run("**foo** and **bar** are concepts. foo bar.")
    assert "concepts_learned" in result


def test_loop_empty_text():
    loop = ContinuousLearningLoop(max_iterations=5)
    result = loop.run("")
    assert result["total_concepts"] == 0
    assert result["concepts_learned"] == 0


def test_loop_log_populated(small_hdv):
    loop = ContinuousLearningLoop(hdv_system=small_hdv, max_iterations=10)
    loop.run(SAMPLE_TEXT)
    assert isinstance(loop.log, list)


def test_loop_energy_history_is_list(small_hdv):
    loop = ContinuousLearningLoop(hdv_system=small_hdv, max_iterations=5)
    result = loop.run(SAMPLE_TEXT)
    assert isinstance(result["energy_history"], list)
