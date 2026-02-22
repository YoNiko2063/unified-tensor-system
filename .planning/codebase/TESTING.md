# Testing Patterns

**Analysis Date:** 2026-02-22

## Test Framework

**Runner:**
- pytest 7.4+ (from `pyproject.toml` line 28)
- Config: `pyproject.toml` [`tool.pytest.ini_options`] (lines 37-39)
- Test paths: `["tests", "platform/backend/tests"]`

**Assertion Library:**
- pytest's built-in assertions: `assert condition`, `assert value == expected`
- Context manager assertions: `pytest.raises(ExceptionType)`
- Custom assertions common in domain tests

**Run Commands:**
```bash
# Run all tests from project root
python -m pytest tests/ -q

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_koopman_edmd.py -q

# Run specific test class
python -m pytest tests/test_code_gen_experiment.py::TestClassifierPredictions -v

# Run with timeout (pytest-timeout plugin, 2.1+)
python -m pytest tests/ --timeout=10

# Platform backend tests
python -m pytest platform/backend/tests/ -q
```

## Test File Organization

**Location:**
- Co-located: `/home/nyoo/projects/unified-tensor-system/tests/` for core tests
- Backend: `/home/nyoo/projects/unified-tensor-system/platform/backend/tests/` for API tests
- Each test file mirrors module structure: `tests/test_X.py` ↔ `tensor/X.py` or `optimization/X.py`

**Naming:**
- Test files: `test_<module>.py` (e.g., `test_koopman_edmd.py`, `test_code_gen_experiment.py`)
- Test functions: `test_<feature_under_test>()` or `test_<class>_<behavior>()`
- Test classes: `Test<ComponentName>` (e.g., `TestObservableBasis`, `TestFitting`)

**Structure:**
```
tests/
├── test_code_gen_experiment.py     # Tests for optimization/code_gen_experiment.py
├── test_koopman_edmd.py            # Tests for tensor/koopman_edmd.py
├── test_simulation_trainer.py      # Tests for tensor/simulation_trainer.py
├── test_cross_domain_transfer.py   # Integration tests for multi-module features
├── test_tracks_abc.py              # Multi-level integration tests
└── ...
```

## Test Structure

**Suite Organization** (from `tests/test_koopman_edmd.py`):
```python
# ------------------------------------------------------------------
# Fixtures: Linear and nonlinear systems
# ------------------------------------------------------------------

def linear_system_trajectory(T: int = 100, dt: float = 0.05) -> np.ndarray:
    """Helper to generate trajectory data."""
    ...

@pytest.fixture
def linear_traj():
    return linear_system_trajectory(100)


# ------------------------------------------------------------------
# Tests: Observable basis
# ------------------------------------------------------------------

class TestObservableBasis:
    def test_degree1_shape(self):
        """Test observable basis dimension for degree=1."""
        ...
```

**Patterns:**
- Section comments with dashes: `# ── Fixtures ──` for grouping
- Fixtures declared with `@pytest.fixture` decorator
- Test classes group related tests: `TestObservableBasis`, `TestFitting`, `TestEigendecomposition`
- Methods within classes: `def test_<behavior>(self):`
- Standalone test functions also used (not always class-based)

**Root conftest.py** (from `/conftest.py`):
- Handles `sys.path` setup for project imports
- Contains `pytest_collectstart()` hook to re-pin project root before each test file collection
- Ensures `tensor/*`, `optimization/*`, `codegen/*` are importable without per-test boilerplate

**Per-test sys.path setup** (from `tests/test_code_gen_experiment.py`):
```python
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pytest
from optimization.code_gen_experiment import (...)
```

This pattern is common but redundant with root conftest; kept for local clarity.

## Test Structure Examples

**Fixture scope patterns** (from `tests/test_code_gen_experiment.py`):
```python
@pytest.fixture(scope="module")
def clf_scaler():
    """Load classifier once per test module (expensive operation)."""
    return load_classifier(METRICS_JSONL)

@pytest.fixture(scope="module")
def results():
    """Run full experiment once, reuse across tests."""
    return run_experiment(METRICS_JSONL)

@pytest.fixture
def summary(results):  # default scope="function" — depends on module-scoped fixture
    """Recomputed per test from shared results."""
    return compute_summary(results)
```

**Class-based test organization** (from `tests/test_koopman_edmd.py`):
```python
class TestObservableBasis:
    def test_degree1_shape(self):
        k = EDMDKoopman(observable_degree=1)
        x = np.array([1.0, 2.0])
        psi = k.build_observable_basis(x)
        assert psi.shape == (3,)  # 1 constant + 2 linear

class TestFitting:
    def test_fit_trajectory(self, linear_traj):
        k = EDMDKoopman(observable_degree=2)
        k.fit_trajectory(linear_traj)
        assert k._fitted
```

**Helper methods in test class** (from `tests/test_simulation_trainer.py`):
```python
class TestSimulationResult:
    def _make_rc_result(self):
        """Helper to construct test fixture."""
        from tensor.simulation_trainer import SimulationResult
        return SimulationResult(
            circuit_type="rc",
            params={"R": 1000.0, "C": 1e-9},
            eigenvalues=np.array([-1e6]),
            stable=True,
            ...
        )

    def test_rc_text_has_eigenvalue(self):
        r = self._make_rc_result()
        text = r.to_text_description()
        assert "eigenvalue" in text
```

## Mocking

**Framework:** `unittest.mock` (standard library)

**Patterns:**
```python
from unittest.mock import MagicMock, patch

# Patch and yield for context manager (from platform/backend/tests/test_api_routes.py)
with patch("routers.regime.LCAPatchDetector", return_value=mock_detector):
    from main import app
    yield TestClient(app)

# Mock return values
mock_detector = MagicMock()
mock_detector.classify_trajectory.return_value = [mock_patch_result] * 10
```

**What to Mock:**
- External API calls: fetchers, GitHub API, external data sources
- Heavy computations: EDMD eigendecomposition in unit tests (use synthetic data instead)
- Platform dependencies: circuit compilers, Rust binaries (use design ground truth)
- Optional imports: guard model imports with `try/except` + mock fallback

**What NOT to Mock:**
- Core mathematical functions (numpy/scipy operations)
- Data transformations (should test actual output)
- Actual finite-element/spectral computations (use small, fast test cases)
- Dataclass instantiation

**Example: Platform API test** (from `platform/backend/tests/test_api_routes.py` lines 34-49):
```python
@pytest.fixture(scope="module")
def client():
    """Create FastAPI TestClient with LCAPatchDetector mocked."""
    mock_patch_result = MagicMock()
    mock_patch_result.patch_type = "lca"
    mock_patch_result.commutator_norm = 0.01

    mock_detector = MagicMock()
    mock_detector.classify_trajectory.return_value = [mock_patch_result] * 10

    with patch("routers.regime.LCAPatchDetector", return_value=mock_detector):
        from main import app
        yield TestClient(app)
```

## Fixtures and Factories

**Test Data:**
```python
def linear_system_trajectory(T: int = 100, dt: float = 0.05) -> np.ndarray:
    """Synthetic trajectory from stable linear system: ẋ = Ax, A = [[-0.5, 1], [-1, -0.5]]"""
    A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    x = np.array([1.0, 0.0])
    traj = [x.copy()]
    for _ in range(T - 1):
        x = x + dt * (A @ x)
        traj.append(x.copy())
    return np.array(traj)

def nonlinear_trajectory(T: int = 100, dt: float = 0.05) -> np.ndarray:
    """RLC+diode nonlinear trajectory."""
    R, C, L, alpha = 1.0, 1.0, 1.0, 0.3
    x = np.array([1.5, 0.0])
    traj = [x.copy()]
    for _ in range(T - 1):
        # RLC dynamics
        ...
    return np.array(traj)
```

**Location:**
- Inline in test files (no central fixtures/ directory)
- Shared fixtures in conftest.py (rare; mostly in individual test files)
- Helper functions prefixed with underscore: `_make_rc_result()`, `_make_rlc_result()`

**Factory pattern** (from `tests/test_cross_domain_transfer.py` lines 49-60):
```python
def _train_spring_mass_memory() -> KoopmanExperienceMemory:
    """Run spring-mass optimiser on [500, 1000, 1500] Hz; return shared memory."""
    sm_mapper = SpringMassDesignMapper(hdv_dim=64, seed=7)
    sm_evaluator = SpringMassEvaluator(max_Q=10.0, max_energy_loss=0.5)
    memory = KoopmanExperienceMemory()
    for target in _TRAIN_TARGETS:
        opt = SpringMassOptimizer(
            sm_mapper, sm_evaluator, memory,
            n_iter=_SM_ITERS, seed=0,
        )
        opt.optimize(target)
    return memory

@pytest.fixture(scope="module")
def sm_trained_memory():
    return _train_spring_mass_memory()
```

## Coverage

**Requirements:** None enforced by CI/configuration

**View Coverage:**
- Not configured in `pyproject.toml`
- Can run manually: `pytest --cov=tensor --cov=optimization --cov=codegen`
- No coverage target or reporting configured

## Test Types

**Unit Tests:**
- Scope: Single function or class method
- Example: `test_degree1_shape()` tests `EDMDKoopman.build_observable_basis()` shape
- Approach: Synthetic small data, fast execution (< 100ms typical)
- Files: `tests/test_koopman_edmd.py`, `tests/test_simulation_trainer.py`

**Integration Tests:**
- Scope: Multiple modules or subsystems
- Example: `test_rlc_warm_from_spring_mass_beats_cold()` tests optimizer memory + RLC evaluator
- Approach: Realistic data, allows ~10-30 second runtime per test
- Files: `tests/test_cross_domain_transfer.py`, `tests/test_multiobjective_transfer.py`

**Platform/API Tests:**
- Scope: HTTP endpoint behavior
- Framework: `fastapi.testclient.TestClient`
- Approach: Mock heavy computation, test routing and response structure
- Files: `platform/backend/tests/test_api_routes.py`, `platform/backend/tests/test_circuit_route.py`

**E2E Tests:**
- Not detected in codebase
- Smoke tests present: `tests/test_smoke_stability_engine.py`

## Common Patterns

**Async Testing:**
- Not used (no async/await in core tensor/optimization modules)
- Platform backend is async (FastAPI) but tests use synchronous TestClient

**Error Testing** (from `tests/test_koopman_edmd.py`):
```python
def test_unfit_raises(self):
    """Unfitted EDMD should raise RuntimeError on eigendecomposition."""
    k = EDMDKoopman()
    with pytest.raises(RuntimeError):
        k.eigendecomposition()

def test_empty_pairs_raises(self):
    """Empty pairs list should raise ValueError or Exception."""
    k = EDMDKoopman()
    with pytest.raises((ValueError, Exception)):
        k.fit([])
```

**Parameterized Tests:**
- Not heavily used (no `@pytest.mark.parametrize`)
- Instead: manual loops within test class or separate test methods per case
- Example: `test_template_a_predicted_ok()`, `test_template_b_predicted_ok()`, `test_template_c_predicted_fail()` (3 separate tests, not parametrized)

**Floating-point Comparisons:**
- Absolute tolerance: `assert abs(value - expected) < 0.01`
- Relative tolerance: `assert abs(r.natural_freq - expected_f0) / expected_f0 < 0.01`
- NumPy arrays: `np.allclose(a, b, atol=1e-6)`

**Diagnostic Output** (from `tests/test_cross_domain_transfer.py`):
```python
def test_spring_mass_memory_accumulates(sm_trained_memory):
    n = len(sm_trained_memory)
    assert n >= 1, (
        f"Spring-mass memory is empty — Koopman fits failed for all targets."
    )
    print(f"\n  Spring-mass memory: {sm_trained_memory.summary()}")
    for e in sm_trained_memory._entries:
        inv = e.invariant
        print(f"    log_ω₀_norm={inv.log_omega0_norm:.4f}  Q_norm={inv.log_Q_norm:.4f}  "
              f"ζ={inv.damping_ratio:.4f}  domain={e.experience.domain}")
```

## Special Considerations

**sys.path Management:**
- Root conftest.py (`/conftest.py`) pins project root via `pytest_collectstart()` hook
- This allows clean imports: `from tensor.X import Y` without per-file setup
- Some test files still add redundant sys.path setup for clarity (backward compatible)

**ecemath subdependency:**
- ecemath/src added to sys.path by `tensor/core.py` import
- Root conftest re-pins project root afterward to prevent shadowing
- Backend tests explicitly manage this: lines 17-20 of `platform/backend/tests/test_api_routes.py`

**Module scope fixtures:**
- Expensive operations (Koopman fits, classifier training) use `@pytest.fixture(scope="module")`
- Reused across all tests in the module (faster test suite)
- See `tests/test_code_gen_experiment.py` lines 40-52

**Defensive imports:**
- Many modules guard heavy imports with try/except
- Tests don't need to mock these (import errors handled gracefully)
- Example: `tensor/meta_optimizer.py` has multiple `try/except ImportError` blocks

---

*Testing analysis: 2026-02-22*
