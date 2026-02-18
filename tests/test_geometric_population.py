"""Tests for tensor/geometric_population.py"""

import pytest
import numpy as np

from tensor.geometric_population import (
    ExprNode,
    LatexTreeParser,
    GeometricHDVPopulator,
    _all_nodes,
)


# ── ExprNode ──────────────────────────────────────────────────────────────────

def test_leaf_depth_zero():
    n = ExprNode("var", "x")
    assert n.depth == 0

def test_leaf_size_one():
    n = ExprNode("var", "x")
    assert n.size == 1

def test_tree_depth():
    child = ExprNode("var", "x")
    parent = ExprNode("op", "+", [child])
    assert parent.depth == 1

def test_tree_size():
    children = [ExprNode("var", str(i)) for i in range(3)]
    parent = ExprNode("group", "", children)
    assert parent.size == 4  # parent + 3 children

def test_branching_leaf():
    assert ExprNode("var").branching == 0.0

def test_branching_nonzero():
    parent = ExprNode("cmd", "\\frac", [ExprNode("var"), ExprNode("var")])
    assert parent.branching > 0.0


# ── LatexTreeParser ───────────────────────────────────────────────────────────

@pytest.fixture
def parser():
    return LatexTreeParser()

def test_parse_simple(parser):
    node = parser.parse("x + y")
    assert node is not None

def test_parse_returns_node(parser):
    node = parser.parse("a = b")
    assert isinstance(node, ExprNode)

def test_parse_frac(parser):
    node = parser.parse(r"\frac{x}{y}")
    assert node.kind == "cmd"
    assert len(node.children) == 2

def test_parse_single_var(parser):
    node = parser.parse("x")
    assert node.kind == "var"
    assert node.value == "x"

def test_parse_nested(parser):
    node = parser.parse(r"\frac{\partial u}{\partial t}")
    assert node is not None

def test_parse_group(parser):
    node = parser.parse("(a + b)")
    assert node is not None

def test_parse_empty(parser):
    node = parser.parse("")
    assert node is not None

def test_tokenize(parser):
    toks = parser._tokenize("x + 1")
    assert "x" in toks
    assert "+" in toks
    assert "1" in toks


# ── GeometricHDVPopulator ─────────────────────────────────────────────────────

@pytest.fixture
def small_hdv(tmp_path):
    from tensor.integrated_hdv import IntegratedHDVSystem
    return IntegratedHDVSystem(
        hdv_dim=300, n_modes=5, embed_dim=32,
        library_path=str(tmp_path / "lib.json"),
    )

@pytest.fixture
def pop(small_hdv):
    return GeometricHDVPopulator(hdv_system=small_hdv)

@pytest.fixture
def pop_no_hdv():
    return GeometricHDVPopulator(hdv_system=None)

HEAT_EQ = r"\frac{\partial T}{\partial t} = \alpha \nabla^2 T"
SCHRODINGER = r"i \partial_t \psi = H \psi"
SIMPLE = "x + y = z"

def test_populate_returns_array(pop):
    vec = pop.populate_from_latex(HEAT_EQ)
    assert isinstance(vec, np.ndarray)

def test_populate_correct_dim(pop, small_hdv):
    vec = pop.populate_from_latex(HEAT_EQ)
    assert vec.shape == (small_hdv.hdv_dim,)

def test_populate_no_hdv_returns_array(pop_no_hdv):
    vec = pop_no_hdv.populate_from_latex(SIMPLE)
    assert isinstance(vec, np.ndarray)

def test_populate_increments_stored(pop):
    assert pop.n_stored == 0
    pop.populate_from_latex(SIMPLE)
    assert pop.n_stored == 1

def test_populate_batch(pop):
    vecs = pop.populate_batch([HEAT_EQ, SCHRODINGER, SIMPLE])
    assert len(vecs) == 3
    for v in vecs:
        assert isinstance(v, np.ndarray)

def test_vectors_not_zero(pop):
    vec = pop.populate_from_latex(HEAT_EQ)
    assert np.any(vec > 0)

def test_vectors_bounded(pop):
    vec = pop.populate_from_latex(HEAT_EQ)
    assert vec.min() >= 0.0
    assert vec.max() <= 1.0

def test_deterministic(pop):
    v1 = pop.populate_from_latex(SIMPLE)
    v2 = pop.populate_from_latex(SIMPLE)
    # Same equation → same features → same HDV
    assert np.allclose(v1, v2)

def test_different_structures_differ(pop):
    v1 = pop.populate_from_latex("x")
    v2 = pop.populate_from_latex(HEAT_EQ)
    # Different structures → different vectors
    assert not np.allclose(v1, v2)

def test_find_similar_structure(pop):
    pop.populate_from_latex(HEAT_EQ)
    pop.populate_from_latex(SCHRODINGER)
    results = pop.find_similar_structure(HEAT_EQ, top_k=2)
    assert len(results) <= 2

def test_find_similar_returns_list(pop):
    pop.populate_from_latex(SIMPLE)
    results = pop.find_similar_structure("x + z", top_k=5)
    assert isinstance(results, list)

def test_find_similar_has_similarity_key(pop):
    pop.populate_from_latex(SIMPLE)
    results = pop.find_similar_structure("x + z", top_k=1)
    if results:
        assert "similarity" in results[0]
        assert "latex" in results[0]

def test_find_similar_sorted_desc(pop):
    pop.populate_batch([SIMPLE, HEAT_EQ, SCHRODINGER])
    results = pop.find_similar_structure(SIMPLE, top_k=3)
    if len(results) >= 2:
        assert results[0]["similarity"] >= results[1]["similarity"]

def test_hdv_registered_in_geometric_domain(pop, small_hdv):
    pop.populate_from_latex(HEAT_EQ)
    assert "geometric" in small_hdv.domain_masks

def test_extract_features(pop):
    node = pop.parser.parse(HEAT_EQ)
    features = pop._extract_features(node)
    assert "depth" in features
    assert "size" in features
    assert "branching" in features
    assert "var_ratio" in features

def test_features_depth_positive(pop):
    node = pop.parser.parse(HEAT_EQ)
    features = pop._extract_features(node)
    assert features["depth"] >= 0.0

def test_all_nodes(pop):
    node = pop.parser.parse("x + y")
    nodes = _all_nodes(node)
    assert len(nodes) >= 1
