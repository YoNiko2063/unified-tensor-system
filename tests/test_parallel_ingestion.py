"""Tests for tensor/parallel_ingestion.py"""

import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tensor.parallel_ingestion import ParallelPaperIngester


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_hdv(tmp_path):
    from tensor.integrated_hdv import IntegratedHDVSystem
    return IntegratedHDVSystem(
        hdv_dim=300, n_modes=5, embed_dim=32,
        library_path=str(tmp_path / "lib.json"),
    )

def _mock_process(self, paper_id):
    """Fake _process_paper: returns synthetic result without HTTP."""
    hdv_dim = self.hdv.hdv_dim if self.hdv else 300
    return {
        "paper_id": paper_id,
        "equations": [f"dx/dt = -{paper_id}"],
        "hdv_vec": np.random.rand(hdv_dim).astype(np.float32),
        "n_equations": 1,
    }


# ── Constructor / properties ──────────────────────────────────────────────────

def test_default_num_workers():
    ing = ParallelPaperIngester()
    assert ing.num_workers == 4

def test_custom_num_workers():
    ing = ParallelPaperIngester(num_workers=2)
    assert ing.num_workers == 2

def test_initial_equation_count():
    ing = ParallelPaperIngester()
    assert ing.total_equations_ingested == 0

def test_initial_energy():
    ing = ParallelPaperIngester()
    assert ing.current_energy == pytest.approx(1.0)

def test_energy_history_empty():
    ing = ParallelPaperIngester()
    assert ing.energy_history == []


# ── ingest_batch (mocked) ─────────────────────────────────────────────────────

def test_ingest_batch_returns_list(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=2)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        results = ing.ingest_batch(["2301.00001", "2301.00002"])
    assert isinstance(results, list)

def test_ingest_batch_correct_count(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=2)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        results = ing.ingest_batch(["a", "b", "c"])
    assert len(results) == 3

def test_ingest_batch_updates_equation_count(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=2)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        ing.ingest_batch(["p1", "p2"])
    assert ing.total_equations_ingested == 2

def test_ingest_batch_empty(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        results = ing.ingest_batch([])
    assert results == []

def test_ingest_batch_populates_energy_history(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=2)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        ing.ingest_batch(["p1", "p2"])
    assert len(ing.energy_history) >= 1

def test_ingest_batch_energy_nonneg(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=2)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        ing.ingest_batch(["p1", "p2", "p3"])
    assert ing.current_energy >= 0.0


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_returns_dict(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=2)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        ing.ingest_batch(["p1"])
    s = ing.summary()
    assert isinstance(s, dict)

def test_summary_keys(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=2)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        ing.ingest_batch(["p1"])
    s = ing.summary()
    for key in ("papers_processed", "equations_ingested", "final_energy",
                "energy_delta", "lyapunov_stable"):
        assert key in s

def test_summary_papers_processed(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=2)
    with patch.object(ParallelPaperIngester, "_process_paper", _mock_process):
        ing.ingest_batch(["p1", "p2"])
    assert ing.summary()["papers_processed"] == 2


# ── _learn_from_paper ─────────────────────────────────────────────────────────

def test_learn_from_paper_no_hdv():
    """Should not crash when hdv is None."""
    ing = ParallelPaperIngester(hdv_system=None)
    result = {
        "paper_id": "p1",
        "equations": ["dx/dt = -x"],
        "hdv_vec": np.ones(300, dtype=np.float32),
        "n_equations": 1,
    }
    ing._learn_from_paper(result)  # must not raise

def test_learn_from_paper_null_vec(small_hdv):
    """Should not crash when hdv_vec is None (failed download)."""
    ing = ParallelPaperIngester(hdv_system=small_hdv)
    result = {"paper_id": "p1", "equations": [], "hdv_vec": None, "n_equations": 0}
    ing._learn_from_paper(result)  # must not raise

def test_learn_from_paper_updates_count(small_hdv):
    ing = ParallelPaperIngester(hdv_system=small_hdv)
    vec = np.ones(small_hdv.hdv_dim, dtype=np.float32)
    result = {"paper_id": "p1", "equations": ["eq1", "eq2"], "hdv_vec": vec, "n_equations": 2}
    ing._learn_from_paper(result)
    assert ing.total_equations_ingested == 2


# ── Thread safety ─────────────────────────────────────────────────────────────

def test_concurrent_learn_thread_safe(small_hdv):
    """Multiple threads calling _learn_from_paper concurrently must not corrupt state."""
    ing = ParallelPaperIngester(hdv_system=small_hdv, num_workers=8)
    vec = np.random.rand(small_hdv.hdv_dim).astype(np.float32)

    def worker(n):
        result = {"paper_id": str(n), "equations": ["eq"], "hdv_vec": vec, "n_equations": 1}
        ing._learn_from_paper(result)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert ing.total_equations_ingested == 20


# ── Parallel is faster than serial ───────────────────────────────────────────

def test_parallel_faster_than_serial(small_hdv):
    """4-worker batch should be faster than 1-worker batch for multiple papers."""
    delay = 0.05  # seconds per paper

    def slow_process(self, pid):
        time.sleep(delay)
        return {
            "paper_id": pid, "equations": ["dx/dt=-x"],
            "hdv_vec": np.zeros(small_hdv.hdv_dim, dtype=np.float32),
            "n_equations": 1,
        }

    paper_ids = [str(i) for i in range(8)]

    # Serial (1 worker)
    ing_serial = ParallelPaperIngester(hdv_system=small_hdv, num_workers=1)
    with patch.object(ParallelPaperIngester, "_process_paper", slow_process):
        t0 = time.time()
        ing_serial.ingest_batch(paper_ids)
        serial_time = time.time() - t0

    # Parallel (4 workers)
    ing_parallel = ParallelPaperIngester(hdv_system=small_hdv, num_workers=4)
    with patch.object(ParallelPaperIngester, "_process_paper", slow_process):
        t0 = time.time()
        ing_parallel.ingest_batch(paper_ids)
        parallel_time = time.time() - t0

    # Parallel should be at least 1.5× faster
    assert parallel_time < serial_time * 0.75, (
        f"Expected parallel ({parallel_time:.2f}s) < 0.75 × serial ({serial_time:.2f}s)"
    )
