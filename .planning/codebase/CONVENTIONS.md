# Coding Conventions

**Analysis Date:** 2026-02-22

## Naming Patterns

**Files:**
- Lowercase with underscores: `integrated_hdv.py`, `code_gen_experiment.py`, `semantic_geometry.py`
- Test files follow `test_<module>.py` pattern: `test_koopman_edmd.py`, `test_code_gen_experiment.py`
- Module packages use lowercase: `tensor/`, `optimization/`, `codegen/`

**Functions:**
- Lowercase with underscores for regular functions: `load_classifier()`, `e_borrow()`, `feature_vec()`
- Private/helper functions prefixed with single underscore: `_repin_root()`, `_extract_imports()`, `_make_rc_result()`
- Test methods use descriptive snake_case: `test_three_templates()`, `test_e_borrow_ordering()`, `test_template_c_above_dsep()`

**Classes:**
- PascalCase for all classes: `IntegratedHDVSystem`, `CodeGenPipeline`, `EDMDKoopman`, `BifurcationDetector`
- Dataclasses use same convention: `GenerationResult`, `BootstrapResult`, `SimulationResult`
- Enum classes: `BorrowProfile`, `CalendarPhase`, `BifurcationStatus`

**Variables:**
- Regular variables: lowercase with underscores: `hdv_dim`, `rust_source`, `target_consonance`
- Constants: UPPERCASE with underscores: `D_SEP`, `E_PYTHON`, `WEIGHTS`, `PHI`, `CHANNEL_NAMES`, `CYCLE_PERIODS`
- Module-level constants: UPPERCASE: `_MAX_SEMANTIC_DIM`, `_MAX_TOKENS`, `_GAMMA_MIN`, `_TAU_SEMANTIC`
- Private module vars prefixed with underscore: `_ROOT`, `_ECEMATH_SRC`, `_PROJECT_ROOT`

**Types:**
- Type hints use standard Python typing: `Optional[str]`, `Dict[str, int]`, `List[Tuple[float, ...]]`
- NumPy arrays referenced as `np.ndarray`
- Generic/domain-specific types in docstrings: `BorrowVector`, `KoopmanResult`, `IntentSpec`

## Code Style

**Formatting:**
- No auto-formatter configured (black/ruff/yapf)
- Default Python style: 4-space indentation, line wraps as needed
- Imports sorted: stdlib → third-party → local imports, but not strictly enforced
- `from __future__ import annotations` used in nearly all modules for type hint forward-compatibility

**Linting:**
- No linting config files present (no `.flake8`, `.pylintrc`, `ruff.toml`)
- Convention enforced by code review and project practice, not automation
- Code is generally clean with standard PEP 8 compliance

**Line Length:**
- Generally reasonable (no strict limit enforced)
- Docstrings sometimes exceed 80 chars but remain readable

## Import Organization

**Order:**
1. `from __future__ import annotations` (future imports at top)
2. Standard library imports: `os`, `sys`, `json`, `subprocess`, `ast`, `tempfile`, etc.
3. Third-party imports: `numpy`, `scipy`, `sklearn`, `torch`, `pytest`, `fastapi`, `pydantic`
4. Local project imports: `from tensor.*, from optimization.*, from codegen.*`

**Path Aliases:**
- Project root insertions in sys.path: `sys.path.insert(0, _ROOT)` ensures top-level packages importable
- ecemath sub-package: `sys.path.insert(0, _ECEMATH_SRC)` pins before tensor.core imports
- **sys.path management critical**: conftest.py has `pytest_collectstart()` hook to re-pin project root before each test file (see `/conftest.py` lines 26-28)
- Imports follow: `from tensor.X import Y` or `from optimization.X import Y` directly after sys.path setup

**Example pattern** (from `tests/test_code_gen_experiment.py`):
```python
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pytest
from optimization.code_gen_experiment import (
    EXTRACTOR_BIN, METRICS_JSONL, TEMPLATES, D_SEP, E_PYTHON, ...
)
```

## Error Handling

**Patterns:**
- Try/except blocks used defensively for optional imports and external calls
- Bare `except Exception:` common for graceful degradation in meta-optimizers
- Specific exception handling for known failures: `except ImportError:`, `except ValueError:`, `except KeyboardInterrupt:`
- Tests use `pytest.raises()` context manager for expected exceptions

**Example** (from `tensor/domain_registry.py`):
```python
try:
    # Optional heavy import
except Exception:
    # Graceful fallback, continue execution
```

**Test error patterns** (from `tests/test_koopman_edmd.py`):
```python
def test_unfit_raises(self):
    k = EDMDKoopman()
    with pytest.raises(RuntimeError):
        k.eigendecomposition()
```

## Logging

**Framework:** No centralized logging framework (no `logging` module usage observed)

**Patterns:**
- Console output via `print()` for informational messages
- Test output via pytest assertions + print() for diagnostic info
- Example (from `tests/test_cross_domain_transfer.py` line 80):
  ```python
  print(f"\n  Spring-mass memory: {sm_trained_memory.summary()}")
  ```

## Comments

**When to Comment:**
- High-level algorithm descriptions at module top (docstring, not inline comments)
- Complex mathematical operations with variable meaning: `ω₀ = √(k/m)`
- Non-obvious tensor dimensions/shapes: `# (m, d)` for matrix operations
- Borrow checker rules and Rust compilation constraints explicitly marked

**Style:**
- Inline comments use `#` with space: `# Comment here`
- Section markers use dashes: `# ── Section Name ──────────────────────────`
- Block comments use triple-dash for visual separation in complex code

**Example** (from `tensor/semantic_geometry.py`):
```python
# CRITICAL-1 enforcement:
#   TextKoopmanOperator uses _TextEDMD (defined here), NOT EDMDKoopman
#   from tensor.koopman_edmd...
```

## Docstrings

**JSDoc/TSDoc:** Not used (Python-only project)

**Python Docstrings:** Triple-quote style with structured sections

**Module-level** (top of file):
```python
"""Brief one-liner summary.

Longer description paragraph(s) explaining:
  - Architecture or approach
  - Key algorithms or concepts
  - Critical constraints or gotchas

Usage notes or examples if applicable.
"""
```

**Class docstrings** (immediately after `class` line):
```python
class IntegratedHDVSystem:
    """One-line summary.

    Attributes:
        hdv_dim: HDV dimension
        n_modes: Number of modes
    """
```

**Function docstrings** (after `def` line):
```python
def e_borrow(bv: Tuple[float, ...]) -> float:
    """Scalar borrow energy from 6-component BorrowVector."""
    return float(np.dot(WEIGHTS, bv))
```

**Multi-line function** (from `tensor/semantic_geometry.py` lines 59-77):
```python
def fit(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> "_TextEDMD":
    """
    Fit Koopman matrix from (h_t, h_{t+1}) embedding pairs.
    pairs: list of (h_k, h_{k+1}) tuples, each h ∈ R^d (d ≤ MAX_SEMANTIC_DIM)
    """
```

## Function Design

**Size:** Functions range from single-operation (2-3 lines) to ~50+ lines
- Small utility functions: `e_borrow()`, `feature_vec()` (1-5 lines)
- Medium logic functions: 10-30 lines typical
- Complex operations (fitting, orchestration): 30-100 lines acceptable if well-commented

**Parameters:**
- Dataclasses preferred for multi-parameter functions: `IntentSpec` for code generation parameters
- Single-purpose positional args for simple functions
- Keyword-only args used sparingly (not enforced pattern)
- Type hints on all parameters: `def fit(self, pairs: List[Tuple[...]]) -> SomeType:`

**Return Values:**
- Explicit return type hints in signature: `-> float`, `-> Optional[np.ndarray]`, `-> Tuple[int, int]`
- Single return values most common
- Tuples for multiple logically-grouped returns: `(eigenvalues, eigenvectors)`
- Dataclass returns for complex results: `GenerationResult`, `SimulationResult`
- None returns explicit: `-> None` or `-> Optional[SomeType]`

## Module Design

**Exports:**
- No `__all__` lists enforced; all public names available
- Private helpers use single underscore prefix: `_extract_imports()`
- Import statements directly reference needed items: `from module import ClassName, function_name`

**Barrel Files:**
- Not heavily used in primary source
- Some domain template modules (`codegen/templates/`) act as aggregators
- Example: `codegen/templates/__init__.py` or domain-specific registries

**File Organization:**
- Related classes grouped in single file: `semantic_geometry.py` has `_TextEDMD`, `SemanticJacobianEstimator`, `TextKoopmanOperator`
- Standalone utility functions near top: constants, then helper functions, then main classes
- Test fixtures and helper functions inside test modules (not in separate conftest except at project root)

## Domain-Specific Patterns

**Mathematical constants:**
- Named with descriptive subscripts: `CYCLE_PERIODS`, `HALFLIFE`, `WEIGHTS`
- Documented in context: what domain they apply to (trading, circuits, etc.)

**Koopman/Spectral terminology:**
- `eigenvalues`, `eigenvectors` for spectral outputs
- `K` or `K_matrix` for Koopman operator
- Domain-specific: `ω₀` (natural frequency), `ζ` (damping ratio), `Q` (quality factor)

**Borrow/Rust-specific:**
- `BorrowVector` (6-component tuple): `(B1, B2, B3, B4, B5, B6)`
- `e_borrow()`: scalar energy metric derived from BV
- `D_SEP = 0.43`: separator threshold for safe/unsafe classification

---

*Convention analysis: 2026-02-22*
