"""
Root conftest.py — ensures the project root is on sys.path so that
top-level packages (tensor/, optimization/, etc.) are importable
without per-test sys.path manipulation.

Note: tensor/core.py inserts ecemath/src at sys.path[0] when imported,
which would shadow our optimization/ package. We re-pin the project root
at position 0 before each file is collected via pytest_collectstart.
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)


def _repin_root():
    """Ensure _PROJECT_ROOT is at sys.path[0]."""
    if sys.path and sys.path[0] == _PROJECT_ROOT:
        return
    if _PROJECT_ROOT in sys.path:
        sys.path.remove(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_collectstart(collector):
    """Re-pin project root before each file is collected and imported."""
    _repin_root()
