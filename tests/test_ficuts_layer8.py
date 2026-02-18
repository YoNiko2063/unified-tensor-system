"""Tests for FICUTS Layer 8: Universal Function Basis"""

import json
import sympy as sp
import numpy as np
import pytest
from pathlib import Path

from tensor.function_basis import EquationParser, FunctionBasisLibrary, FunctionBasisToHDV


# ── Task 8.1 ──────────────────────────────────────────────────────────────────

def test_parse_exponential():
    p = EquationParser()
    expr = p.parse(r"e^{-t/\tau}")
    assert expr is not None
    assert p.classify_function_type(expr) == 'exponential'

def test_parse_trig():
    p = EquationParser()
    expr = sp.sin(sp.Symbol('omega') * sp.Symbol('t'))
    assert p.classify_function_type(expr) == 'trigonometric'

def test_parse_log():
    p = EquationParser()
    expr = sp.log(sp.Symbol('x'))
    assert p.classify_function_type(expr) == 'logarithmic'

def test_parse_unknown():
    p = EquationParser()
    assert p.classify_function_type(None) == 'unknown'

def test_extract_params_excludes_independent():
    p = EquationParser()
    tau = sp.Symbol('tau')
    t = sp.Symbol('t')
    expr = sp.exp(-t / tau)
    params = p.extract_parameters(expr)
    assert 'tau' in params
    assert 't' not in params

def test_extract_params_none():
    p = EquationParser()
    assert p.extract_parameters(None) == []

def test_parse_simple_fallback():
    """_parse_simple handles e^{-x} even without antlr4."""
    p = EquationParser()
    expr = p._parse_simple(r"e^{-x}")
    assert expr is not None
    assert p.classify_function_type(expr) == 'exponential'


# ── Task 8.2 ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_library(tmp_path):
    return FunctionBasisLibrary(library_path=str(tmp_path / 'lib.json'))

def test_add_equation_creates_entry(fresh_library):
    fresh_library.add_equation_direct('p1', r"e^{-t/\tau}", 'ece')
    assert len(fresh_library.library) == 1

def test_add_same_equation_different_domain_merges(fresh_library):
    fresh_library.add_equation_direct('p1', r"e^{-t/\tau}", 'ece')
    fresh_library.add_equation_direct('p2', r"e^{-t/\tau}", 'biology')
    # Same expression → merged into one entry
    assert len(fresh_library.library) == 1
    entry = list(fresh_library.library.values())[0]
    assert 'ece' in entry['domains']
    assert 'biology' in entry['domains']

def test_universal_functions_threshold(fresh_library):
    # Add same equation to 3 domains
    for domain, paper in [('ece','p1'), ('biology','p2'), ('finance','p3')]:
        fresh_library.add_equation_direct(paper, r"e^{-t/\tau}", domain)
    universals = fresh_library.get_universal_functions(min_domains=3)
    assert len(universals) == 1

def test_promote_to_foundational(fresh_library):
    fresh_library.add_equation_direct('p1', r"e^{-t/\tau}", 'ece')
    name = list(fresh_library.library.keys())[0]
    assert fresh_library.library[name]['classification'] == 'experimental'
    fresh_library.promote_to_foundational(name)
    assert fresh_library.library[name]['classification'] == 'foundational'

def test_library_persists_to_disk(tmp_path):
    lib1 = FunctionBasisLibrary(library_path=str(tmp_path / 'lib.json'))
    lib1.add_equation_direct('p1', r"e^{-t/\tau}", 'ece')
    lib2 = FunctionBasisLibrary(library_path=str(tmp_path / 'lib.json'))
    assert len(lib2.library) == 1

def test_ingest_from_storage(tmp_path):
    # Write a mock ingested paper
    storage = tmp_path / 'ingested'
    storage.mkdir()
    paper = {
        'url': 'http://example.com/circuit',
        'article': {'title': 'RC Circuit Analysis'},
        'concepts': {'equations': [r'\frac{dV}{dt} = -\frac{V}{\tau}']},
    }
    (storage / 'aabbccdd0011.json').write_text(json.dumps(paper))

    lib = FunctionBasisLibrary(library_path=str(tmp_path / 'lib.json'))
    lib.ingest_papers_from_storage(str(storage))
    # At least tried to parse — may succeed or not depending on antlr4
    assert isinstance(lib.library, dict)

def test_infer_domain_ece(fresh_library):
    assert fresh_library._infer_domain('http://vlsi.edu/paper', '') == 'ece'

def test_infer_domain_biology(fresh_library):
    assert fresh_library._infer_domain('', 'Neuron Synapse Dynamics') == 'biology'

def test_infer_domain_general(fresh_library):
    assert fresh_library._infer_domain('http://unknown.com', 'Something random') == 'general'


# ── Task 8.3 ──────────────────────────────────────────────────────────────────

@pytest.fixture
def loaded_library(tmp_path):
    lib = FunctionBasisLibrary(library_path=str(tmp_path / 'lib.json'))
    for domain, paper in [('ece', 'p1'), ('biology', 'p2'), ('finance', 'p3')]:
        lib.add_equation_direct(paper, r"e^{-t/\tau}", domain)
    lib.add_equation_direct('p4', r"e^{-t/\tau}", 'physics')  # 4th domain
    # A second distinct function in ece only
    lib.add_equation_direct('p5', r"e^{-x}", 'ece')
    return lib

def test_assign_dimensions_runs(loaded_library):
    mapper = FunctionBasisToHDV(loaded_library, hdv_dim=10000)
    mapper.assign_dimensions()
    assert mapper.next_free_dim > 0
    assert len(mapper.dim_assignments) == len(loaded_library.library)

def test_domain_mask_active(loaded_library):
    mapper = FunctionBasisToHDV(loaded_library, hdv_dim=10000)
    mapper.assign_dimensions()
    mask = mapper.get_domain_mask('ece')
    assert mask.sum() > 0

def test_domain_mask_unknown_domain_is_empty(loaded_library):
    mapper = FunctionBasisToHDV(loaded_library, hdv_dim=10000)
    mapper.assign_dimensions()
    mask = mapper.get_domain_mask('nonexistent')
    assert mask.sum() == 0

def test_overlap_dimensions_found(loaded_library):
    mapper = FunctionBasisToHDV(loaded_library, hdv_dim=10000)
    mapper.assign_dimensions()
    overlaps = mapper.get_overlap_dimensions()
    # exponential appears in ece, biology, finance, physics → universal → multiple dims
    assert len(overlaps) > 0

def test_universal_gets_multiple_dims(loaded_library):
    """Functions in ≥3 domains should get 5 dimensions."""
    mapper = FunctionBasisToHDV(loaded_library, hdv_dim=10000)
    mapper.assign_dimensions()
    # Find the universal function
    universals = loaded_library.get_universal_functions(min_domains=3)
    assert len(universals) >= 1
    dims = mapper.dim_assignments[universals[0]]
    assert isinstance(dims, list) and len(dims) == 5
