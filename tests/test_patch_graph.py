"""
Tests for PatchGraph — global topological map of HDVS patches.
"""

import numpy as np
import pytest
from tensor.patch_graph import Patch, PatchGraph
from tensor.lca_patch_detector import PatchClassification


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_patch(patch_id: int, patch_type: str = 'lca') -> Patch:
    """Create a minimal Patch for testing."""
    n = 2
    basis = np.eye(n).reshape(1, n, n)  # 1 × 2 × 2
    spectrum = np.array([-0.5 + 0j, -1.0 + 0j])
    centroid = np.zeros(n)
    return Patch(
        id=patch_id,
        patch_type=patch_type,
        operator_basis=basis,
        spectrum=spectrum,
        centroid=centroid,
        operator_rank=1,
        commutator_norm=0.0,
        curvature_ratio=0.02,
        spectral_gap=0.5,
    )


def make_classification(patch_type: str = 'lca') -> PatchClassification:
    """Create a minimal PatchClassification for testing from_classification."""
    n = 2
    return PatchClassification(
        patch_type=patch_type,
        operator_rank=1,
        commutator_norm=0.0,
        curvature_ratio=0.02,
        spectral_gap=0.5,
        basis_matrices=np.eye(n).reshape(1, n, n),
        eigenvalues=np.array([-0.5 + 0j, -1.0 + 0j]),
        centroid=np.zeros(n),
    )


@pytest.fixture
def empty_graph():
    return PatchGraph()


@pytest.fixture
def triangle_graph():
    """Graph with 3 patches in a triangle."""
    g = PatchGraph()
    pa = make_patch(0, 'lca')
    pb = make_patch(1, 'nonabelian')
    pc = make_patch(2, 'lca')
    g.add_patch(pa)
    g.add_patch(pb)
    g.add_patch(pc)
    g.add_transition(pa, pb, curvature_cost=0.3)
    g.add_transition(pb, pc, curvature_cost=0.2)
    g.add_transition(pa, pc, curvature_cost=0.9)  # expensive direct edge
    return g, pa, pb, pc


# ------------------------------------------------------------------
# Tests: Patch dataclass
# ------------------------------------------------------------------

class TestPatch:
    def test_feature_vector_shape(self):
        p = make_patch(0)
        fv = p.feature_vector()
        # 4 scalars (rank, comm_norm, curv_ratio, gap) + 3 onehot + 4 real + 4 imag = 15
        assert fv.shape == (15,)

    def test_feature_vector_dtype(self):
        p = make_patch(0)
        assert p.feature_vector().dtype == np.float32

    def test_feature_vector_lca_onehot(self):
        p = make_patch(0, 'lca')
        fv = p.feature_vector()
        # type onehot at positions [4, 5, 6]
        assert fv[4] == 1.0  # lca
        assert fv[5] == 0.0
        assert fv[6] == 0.0

    def test_feature_vector_nonabelian_onehot(self):
        p = make_patch(0, 'nonabelian')
        fv = p.feature_vector()
        assert fv[4] == 0.0
        assert fv[5] == 1.0  # nonabelian
        assert fv[6] == 0.0

    def test_feature_vector_chaotic_onehot(self):
        p = make_patch(0, 'chaotic')
        fv = p.feature_vector()
        assert fv[4] == 0.0
        assert fv[5] == 0.0
        assert fv[6] == 1.0  # chaotic

    def test_from_classification(self):
        cl = make_classification('lca')
        p = Patch.from_classification(patch_id=42, classification=cl)
        assert p.id == 42
        assert p.patch_type == 'lca'
        assert p.operator_rank == cl.operator_rank
        assert np.allclose(p.centroid, cl.centroid)

    def test_from_classification_nonabelian(self):
        cl = make_classification('nonabelian')
        p = Patch.from_classification(patch_id=7, classification=cl)
        assert p.patch_type == 'nonabelian'


# ------------------------------------------------------------------
# Tests: Node management
# ------------------------------------------------------------------

class TestNodeManagement:
    def test_add_patch(self, empty_graph):
        p = make_patch(0)
        empty_graph.add_patch(p)
        assert empty_graph.get_patch(0) is p

    def test_get_nonexistent(self, empty_graph):
        assert empty_graph.get_patch(99) is None

    def test_all_patches_empty(self, empty_graph):
        assert empty_graph.all_patches() == []

    def test_all_patches(self, empty_graph):
        p0 = make_patch(0)
        p1 = make_patch(1)
        empty_graph.add_patch(p0)
        empty_graph.add_patch(p1)
        ids = {p.id for p in empty_graph.all_patches()}
        assert ids == {0, 1}

    def test_lca_patches_filter(self, empty_graph):
        p0 = make_patch(0, 'lca')
        p1 = make_patch(1, 'nonabelian')
        p2 = make_patch(2, 'lca')
        empty_graph.add_patch(p0)
        empty_graph.add_patch(p1)
        empty_graph.add_patch(p2)
        lca = empty_graph.lca_patches()
        assert len(lca) == 2
        assert all(p.patch_type == 'lca' for p in lca)

    def test_new_patch_id_unique(self, empty_graph):
        ids = [empty_graph.new_patch_id() for _ in range(5)]
        assert ids == sorted(set(ids))


# ------------------------------------------------------------------
# Tests: Edge management
# ------------------------------------------------------------------

class TestEdgeManagement:
    def test_add_transition(self, empty_graph):
        p0 = make_patch(0)
        p1 = make_patch(1)
        empty_graph.add_patch(p0)
        empty_graph.add_patch(p1)
        empty_graph.add_transition(p0, p1, curvature_cost=0.5)
        neighbors = empty_graph.get_neighbors(0)
        assert (1, 0.5) in neighbors

    def test_symmetric_edge(self, empty_graph):
        """Adding an edge A→B also creates B→A with same cost."""
        p0 = make_patch(0)
        p1 = make_patch(1)
        empty_graph.add_patch(p0)
        empty_graph.add_patch(p1)
        empty_graph.add_transition(p0, p1, curvature_cost=0.4)
        neighbors_1 = empty_graph.get_neighbors(1)
        neighbor_ids = [n[0] for n in neighbors_1]
        assert 0 in neighbor_ids

    def test_keeps_min_cost(self, empty_graph):
        """If edge already exists, keep minimum cost."""
        p0 = make_patch(0)
        p1 = make_patch(1)
        empty_graph.add_patch(p0)
        empty_graph.add_patch(p1)
        empty_graph.add_transition(p0, p1, curvature_cost=0.8)
        empty_graph.add_transition(p0, p1, curvature_cost=0.3)
        neighbors = empty_graph.get_neighbors(0)
        cost = dict(neighbors)[1]
        assert abs(cost - 0.3) < 1e-10

    def test_no_neighbors_for_isolated_node(self, empty_graph):
        p = make_patch(0)
        empty_graph.add_patch(p)
        assert empty_graph.get_neighbors(0) == []


# ------------------------------------------------------------------
# Tests: Shortest path (Dijkstra)
# ------------------------------------------------------------------

class TestShortestPath:
    def test_same_node(self, empty_graph):
        p = make_patch(0)
        empty_graph.add_patch(p)
        path = empty_graph.shortest_path(0, 0)
        assert path == [0]

    def test_direct_edge(self, empty_graph):
        p0 = make_patch(0)
        p1 = make_patch(1)
        empty_graph.add_patch(p0)
        empty_graph.add_patch(p1)
        empty_graph.add_transition(p0, p1, curvature_cost=0.5)
        path = empty_graph.shortest_path(0, 1)
        assert path == [0, 1]

    def test_triangle_prefers_two_hops(self, triangle_graph):
        """Triangle: 0→2 direct (cost 0.9) vs 0→1→2 (cost 0.5). Should choose 0→1→2."""
        g, pa, pb, pc = triangle_graph
        path = g.shortest_path(0, 2)
        # 0→1→2 has cost 0.5 vs 0→2 has cost 0.9
        assert path == [0, 1, 2]

    def test_no_path_returns_empty(self, empty_graph):
        p0 = make_patch(0)
        p1 = make_patch(1)
        empty_graph.add_patch(p0)
        empty_graph.add_patch(p1)
        # No transition added
        path = empty_graph.shortest_path(0, 1)
        assert path == []

    def test_missing_start(self, empty_graph):
        p = make_patch(1)
        empty_graph.add_patch(p)
        path = empty_graph.shortest_path(99, 1)
        assert path == []

    def test_missing_end(self, empty_graph):
        p = make_patch(0)
        empty_graph.add_patch(p)
        path = empty_graph.shortest_path(0, 99)
        assert path == []

    def test_path_includes_endpoints(self, triangle_graph):
        g, pa, pb, pc = triangle_graph
        path = g.shortest_path(0, 2)
        assert path[0] == 0
        assert path[-1] == 2

    def test_path_cost_correct(self, triangle_graph):
        g, pa, pb, pc = triangle_graph
        path = g.shortest_path(0, 2)
        cost = g.path_cost(path)
        assert abs(cost - 0.5) < 1e-10  # 0.3 + 0.2 = 0.5

    def test_longer_chain(self, empty_graph):
        """4-node chain: 0→1→2→3 with equal costs."""
        patches = [make_patch(i) for i in range(4)]
        for p in patches:
            empty_graph.add_patch(p)
        for i in range(3):
            empty_graph.add_transition(patches[i], patches[i + 1], 1.0)
        path = empty_graph.shortest_path(0, 3)
        assert path == [0, 1, 2, 3]


# ------------------------------------------------------------------
# Tests: Path cost
# ------------------------------------------------------------------

class TestPathCost:
    def test_empty_path_cost(self, empty_graph):
        assert empty_graph.path_cost([]) == 0.0

    def test_single_node_cost(self, empty_graph):
        assert empty_graph.path_cost([0]) == 0.0

    def test_missing_edge_is_inf(self, empty_graph):
        # Path references nonexistent edges
        cost = empty_graph.path_cost([0, 99])
        assert cost == float('inf')


# ------------------------------------------------------------------
# Tests: HDV embedding export
# ------------------------------------------------------------------

class TestHDVExport:
    def test_export_returns_dict(self, empty_graph):
        p0 = make_patch(0)
        empty_graph.add_patch(p0)
        result = empty_graph.export_hdv_embedding()
        assert isinstance(result, dict)

    def test_export_keys_match_patches(self, empty_graph):
        p0 = make_patch(0)
        p1 = make_patch(1)
        empty_graph.add_patch(p0)
        empty_graph.add_patch(p1)
        result = empty_graph.export_hdv_embedding()
        assert set(result.keys()) == {0, 1}

    def test_export_values_are_arrays(self, empty_graph):
        p0 = make_patch(0)
        empty_graph.add_patch(p0)
        result = empty_graph.export_hdv_embedding()
        assert isinstance(result[0], np.ndarray)


# ------------------------------------------------------------------
# Tests: Summary
# ------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self, empty_graph):
        s = empty_graph.summary()
        assert s['n_patches'] == 0
        assert s['n_edges'] == 0

    def test_summary_counts(self, triangle_graph):
        g, pa, pb, pc = triangle_graph
        s = g.summary()
        assert s['n_patches'] == 3
        assert s['n_edges'] == 3  # 3 undirected edges

    def test_type_counts(self, triangle_graph):
        g, pa, pb, pc = triangle_graph
        s = g.summary()
        assert s['type_counts']['lca'] == 2
        assert s['type_counts']['nonabelian'] == 1

    def test_lca_fraction(self, triangle_graph):
        g, pa, pb, pc = triangle_graph
        s = g.summary()
        assert abs(s['lca_fraction'] - 2/3) < 1e-10

    def test_repr_has_patch_count(self, triangle_graph):
        g, pa, pb, pc = triangle_graph
        r = repr(g)
        assert 'patches=3' in r


# ------------------------------------------------------------------
# Tests: 3-component edge cost (whattodo.md gaps)
# ------------------------------------------------------------------

class TestThreeComponentEdgeCost:
    def test_combined_cost_curvature_only(self):
        costs = (0.4, 0.0, 0.0)
        assert abs(PatchGraph.combined_cost(costs, 1.0, 0.5, 0.5) - 0.4) < 1e-10

    def test_combined_cost_all_components(self):
        costs = (0.2, 0.4, 0.6)
        # 1.0*0.2 + 0.5*0.4 + 0.5*0.6 = 0.2 + 0.2 + 0.3 = 0.7
        expected = 1.0 * 0.2 + 0.5 * 0.4 + 0.5 * 0.6
        assert abs(PatchGraph.combined_cost(costs) - expected) < 1e-10

    def test_add_transition_three_components(self):
        g = PatchGraph()
        p0 = make_patch(0)
        p1 = make_patch(1)
        g.add_patch(p0)
        g.add_patch(p1)
        g.add_transition(p0, p1, curvature_cost=0.3, interval_cost=0.1, koopman_risk=0.2)
        neighbors = g.get_neighbors(0)
        # combined = 1.0*0.3 + 0.5*0.1 + 0.5*0.2 = 0.3 + 0.05 + 0.1 = 0.45
        expected = 1.0 * 0.3 + 0.5 * 0.1 + 0.5 * 0.2
        costs = dict(neighbors)
        assert abs(costs[1] - expected) < 1e-10

    def test_keeps_min_combined_cost(self):
        """Keep edge with lower combined cost, not just lower curvature."""
        g = PatchGraph()
        p0 = make_patch(0)
        p1 = make_patch(1)
        g.add_patch(p0)
        g.add_patch(p1)
        # First edge: combined = 1.0*0.8 + 0 + 0 = 0.8
        g.add_transition(p0, p1, curvature_cost=0.8)
        # Second edge: combined = 1.0*0.2 + 0 + 0 = 0.2 (cheaper)
        g.add_transition(p0, p1, curvature_cost=0.2)
        neighbors = g.get_neighbors(0)
        cost = dict(neighbors)[1]
        assert abs(cost - 0.2) < 1e-10

    def test_custom_weights_in_constructor(self):
        g = PatchGraph(alpha=2.0, beta=0.0, gamma=0.0)
        p0 = make_patch(0)
        p1 = make_patch(1)
        g.add_patch(p0)
        g.add_patch(p1)
        g.add_transition(p0, p1, curvature_cost=0.5, interval_cost=0.9, koopman_risk=0.9)
        # With beta=0, gamma=0: only curvature matters → 2.0 * 0.5 = 1.0
        neighbors = g.get_neighbors(0)
        cost = dict(neighbors)[1]
        assert abs(cost - 1.0) < 1e-10

    def test_path_cost_with_custom_weights(self):
        g = PatchGraph()
        patches = [make_patch(i) for i in range(3)]
        for p in patches:
            g.add_patch(p)
        g.add_transition(patches[0], patches[1], curvature_cost=0.2, interval_cost=0.4, koopman_risk=0.0)
        g.add_transition(patches[1], patches[2], curvature_cost=0.1, interval_cost=0.2, koopman_risk=0.0)
        path = [0, 1, 2]
        # alpha=1.0, beta=0.5, gamma=0.5
        # edge(0,1): 0.2 + 0.5*0.4 = 0.4; edge(1,2): 0.1 + 0.5*0.2 = 0.2 → total 0.6
        cost = g.path_cost(path)
        expected = (0.2 + 0.5 * 0.4) + (0.1 + 0.5 * 0.2)
        assert abs(cost - expected) < 1e-10

    def test_shortest_path_uses_combined_cost(self):
        """Shortest path chooses lower *combined* cost even if curvature alone differs."""
        g = PatchGraph()
        # 0→2 direct: curvature=0.1, interval=0.0 → combined=0.1
        # 0→1→2: curvature=0.05+0.05=0.1, interval=0.4+0.4=0.8 → combined=0.1+0.5*0.8=0.5
        patches = [make_patch(i) for i in range(3)]
        for p in patches:
            g.add_patch(p)
        g.add_transition(patches[0], patches[2], curvature_cost=0.1, interval_cost=0.0)
        g.add_transition(patches[0], patches[1], curvature_cost=0.05, interval_cost=0.4)
        g.add_transition(patches[1], patches[2], curvature_cost=0.05, interval_cost=0.4)
        path = g.shortest_path(0, 2)
        # Direct [0,2] has cost 0.1; indirect [0,1,2] has cost 0.5 → direct wins
        assert path == [0, 2]
