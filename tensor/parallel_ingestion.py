"""
FICUTS Parallel Paper Ingestion

Downloads and processes multiple arXiv papers concurrently.
Each worker thread owns its IntegratedHDVSystem instance (no lock on encoding).
Results are merged into a shared HDV via superposition — thread-safe via RLock.

Learning happens DURING ingestion: each completed paper immediately updates
the main HDV system so subsequent papers benefit from prior discoveries.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np


class ParallelPaperIngester:
    """
    Ingest arXiv papers in parallel.

    Architecture:
      - num_workers threads, each with a private IntegratedHDVSystem
      - Equations encoded locally (no shared state during compute)
      - Results superposed into main hdv_system under a lock
      - Lyapunov-like energy tracked: energy decreases as more papers ingested

    Usage:
      ingester = ParallelPaperIngester(hdv_system=hdv, num_workers=4)
      results = ingester.ingest_batch(["2301.00001", "2301.00002", ...])
      print(ingester.total_equations_ingested)
    """

    def __init__(self, hdv_system=None, num_workers: int = 4,
                 rate_limit_seconds: float = 1.0):
        self.hdv = hdv_system
        self.num_workers = num_workers
        self.rate_limit_seconds = rate_limit_seconds

        self._lock = threading.RLock()
        self._results: List[Dict] = []
        self._equation_count = 0
        self._energy: float = 1.0  # starts high, decreases with learning
        self._energy_history: List[float] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_batch(self, paper_ids: List[str]) -> List[Dict]:
        """
        Process papers concurrently.

        Returns list of result dicts:
          {paper_id, equations, hdv_vec, n_equations, error?}

        Learning (HDV update + energy update) happens as each paper completes.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_paper, pid): pid
                for pid in paper_ids
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self._learn_from_paper(result)
                    results.append(result)

        return results

    def ingest_directory(self, directory: str, pattern: str = "*.tar.gz") -> List[Dict]:
        """
        Ingest all papers found in a local directory (pre-downloaded).

        Useful when arXiv rate limits prevent live downloads.
        """
        from pathlib import Path
        paths = list(Path(directory).glob(pattern))
        if not paths:
            return []

        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_local_file, str(p)): p
                for p in paths
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self._learn_from_paper(result)
                    results.append(result)

        return results

    @property
    def total_equations_ingested(self) -> int:
        return self._equation_count

    @property
    def current_energy(self) -> float:
        """Lyapunov-like energy: decreases as more unique equations ingested."""
        return self._energy

    @property
    def energy_history(self) -> List[float]:
        return list(self._energy_history)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "papers_processed": len(self._results),
                "equations_ingested": self._equation_count,
                "final_energy": self._energy,
                "energy_delta": (
                    self._energy_history[0] - self._energy_history[-1]
                    if len(self._energy_history) >= 2 else 0.0
                ),
                "lyapunov_stable": (
                    len(self._energy_history) < 2
                    or self._energy_history[-1] <= self._energy_history[-2]
                ),
            }

    # ── Worker methods (run in threads) ───────────────────────────────────────

    def _process_paper(self, paper_id: str) -> Optional[Dict]:
        """Download + parse + encode one paper (runs in worker thread)."""
        try:
            from tensor.arxiv_pdf_parser import ArxivPDFSourceParser
            from tensor.integrated_hdv import IntegratedHDVSystem

            # Each thread owns its HDV — no lock needed during encoding
            hdv_dim = self.hdv.hdv_dim if self.hdv else 10000
            n_modes = self.hdv.n_modes if self.hdv else 150
            embed_dim = self.hdv.embed_dim if self.hdv else 512

            local_hdv = IntegratedHDVSystem(
                hdv_dim=hdv_dim, n_modes=n_modes, embed_dim=embed_dim,
            )
            parser = ArxivPDFSourceParser(
                rate_limit_seconds=self.rate_limit_seconds
            )
            equations = parser.extract_equations_from_paper(paper_id)

            # Encode and superpose locally
            hdv_vecs = [
                local_hdv.encode_equation(eq, "math") for eq in equations
            ]
            merged = (
                np.clip(sum(hdv_vecs), 0.0, 1.0)
                if hdv_vecs
                else np.zeros(hdv_dim, dtype=np.float32)
            )

            return {
                "paper_id": paper_id,
                "equations": equations,
                "hdv_vec": merged,
                "n_equations": len(equations),
            }

        except Exception as exc:
            return {
                "paper_id": paper_id,
                "equations": [],
                "hdv_vec": None,
                "n_equations": 0,
                "error": str(exc),
            }

    def _process_local_file(self, path: str) -> Optional[Dict]:
        """Parse a locally downloaded arXiv source file."""
        try:
            from tensor.arxiv_pdf_parser import ArxivPDFSourceParser
            from tensor.integrated_hdv import IntegratedHDVSystem

            hdv_dim = self.hdv.hdv_dim if self.hdv else 10000
            local_hdv = IntegratedHDVSystem(hdv_dim=hdv_dim, n_modes=10, embed_dim=64)

            parser = ArxivPDFSourceParser(rate_limit_seconds=0)
            equations = parser._extract_from_local_file(path)

            hdv_vecs = [
                local_hdv.encode_equation(eq, "math") for eq in equations
            ]
            merged = (
                np.clip(sum(hdv_vecs), 0.0, 1.0)
                if hdv_vecs
                else np.zeros(hdv_dim, dtype=np.float32)
            )

            return {
                "paper_id": path,
                "equations": equations,
                "hdv_vec": merged,
                "n_equations": len(equations),
            }

        except Exception as exc:
            return {
                "paper_id": path, "equations": [],
                "hdv_vec": None, "n_equations": 0, "error": str(exc),
            }

    # ── Learning (thread-safe) ────────────────────────────────────────────────

    def _learn_from_paper(self, result: Dict) -> None:
        """
        Merge paper's HDV into main system and update Lyapunov energy.

        Energy update: E_{n+1} = E_n · (1 - Δn / total_dims)
        where Δn = new dimensions activated by this paper.
        """
        if self.hdv is None or result.get("hdv_vec") is None:
            return

        vec = result["hdv_vec"]
        n_eq = result.get("n_equations", 0)

        with self._lock:
            # Count dims newly activated
            old_active = np.sum(
                self.hdv.domain_masks.get("math", np.zeros(self.hdv.hdv_dim, dtype=bool))
            )
            self.hdv._register_domain_dims("math", vec)
            new_active = np.sum(self.hdv.domain_masks.get("math", np.zeros(1)))
            delta = int(new_active - old_active)

            self._equation_count += n_eq
            self._results.append(result)

            # Lyapunov update: more unique dims → energy decreases
            if self.hdv.hdv_dim > 0:
                self._energy = max(
                    0.0,
                    self._energy * (1.0 - delta / self.hdv.hdv_dim)
                )
            self._energy_history.append(self._energy)
