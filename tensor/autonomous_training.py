"""
FICUTS Phase 4: Autonomous Learning System

Orchestrates all 5 learning dimensions simultaneously via threaded workers.

Threads:
  math-arxiv       : process ingested arXiv papers → extract equations → math HDVs
  math-library     : keep FunctionBasisLibrary synced → reassign HDV dims
  behavioral-dw    : DeepWiki workflow ingestion → behavioral HDVs
  behavioral-gh    : GitHub API fallback for repos DeepWiki doesn't cover
  discovery        : scan for cross-dimensional universals every 60s
  (network-trainer : future — train UnifiedTensorNetwork on accumulated patterns)

Usage:
  from tensor.autonomous_training import run_autonomous_learning
  run_autonomous_learning()          # runs until Ctrl+C

Or programmatically:
  system = AutonomousLearningSystem()
  system.start(repos=['https://github.com/...'])
  # ... do other work ...
  system.stop()
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from tensor.integrated_hdv import IntegratedHDVSystem
from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery
from tensor.arxiv_pdf_parser import ArxivPDFSourceParser
from tensor.function_basis import FunctionBasisLibrary, EquationParser
from tensor.deepwiki_integration import DeepWikiWorkflowParser
from tensor.github_api_fallback import GitHubAPICapabilityExtractor


_DEFAULT_REPOS = [
    "https://github.com/huggingface/transformers",
    "https://github.com/pytorch/pytorch",
    "https://github.com/numpy/numpy",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/Ultimaker/Cura",
    "https://github.com/prusa3d/PrusaSlicer",
    "https://github.com/langchain-ai/langchain",
    "https://github.com/matplotlib/matplotlib",
]


class AutonomousLearningSystem:
    """
    Orchestrates all learning dimensions via background threads.

    All threads communicate through:
    - IntegratedHDVSystem  : shared HDV space (read-heavy, thread-safe reads)
    - CrossDimensionalDiscovery : accumulates patterns across all threads
    - self._lock           : protects stats dict writes
    """

    def __init__(
        self,
        hdv_dim: int = 10000,
        n_modes: int = 150,
        embed_dim: int = 512,
        github_token: Optional[str] = None,
        ingested_dir: str = "tensor/data/ingested",
    ):
        self.ingested_dir = ingested_dir

        # Core systems
        self.hdv_system = IntegratedHDVSystem(hdv_dim, n_modes, embed_dim)
        self.discovery = CrossDimensionalDiscovery(self.hdv_system)

        # Dimension workers
        self.arxiv_parser = ArxivPDFSourceParser(rate_limit_seconds=1.5)
        self.function_library = FunctionBasisLibrary()
        self.eq_parser = EquationParser()
        self.deepwiki = DeepWikiWorkflowParser(rate_limit_seconds=2.0)
        self.github = GitHubAPICapabilityExtractor(token=github_token)

        # Threading
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []

        # Stats (protected by _lock)
        self.stats: Dict = {
            "math_patterns": 0,
            "behavioral_patterns": 0,
            "universals_found": 0,
            "start_time": None,
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, repos: Optional[List[str]] = None):
        """Start all learning threads."""
        self._stop_event.clear()
        self.stats["start_time"] = time.time()

        target_repos = repos or _DEFAULT_REPOS

        thread_defs = [
            ("math-arxiv",       self._math_arxiv_thread,        ()),
            ("math-library",     self._math_library_thread,      ()),
            ("behavioral-dw",    self._behavioral_deepwiki_thread, (target_repos,)),
            ("discovery",        self._discovery_thread,          ()),
        ]

        for name, fn, args in thread_defs:
            t = threading.Thread(target=fn, args=args, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            print(f"[ALS] Started: {name}")

        print(f"[ALS] System running. {len(target_repos)} repos queued. "
              f"Ctrl+C to stop.")

    def stop(self):
        """Graceful shutdown — signal threads, wait, save state."""
        print("[ALS] Shutting down...")
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()

        # Persist
        self.discovery.save_universals()
        self.hdv_system.save_state()

        u_count = len(self.discovery.universals)
        print(f"[ALS] Shutdown complete. Universals: {u_count}. "
              f"State saved.")

    def print_status(self):
        """Print current system status."""
        with self._lock:
            stats = dict(self.stats)

        start = stats.get("start_time") or time.time()
        uptime_h = (time.time() - start) / 3600
        counts = self.discovery.get_pattern_counts()

        print(
            f"\n[ALS Status @ {uptime_h:.2f}h]\n"
            f"  Math patterns:      {stats['math_patterns']}\n"
            f"  Behavioral patterns:{stats['behavioral_patterns']}\n"
            f"  Universals found:   {stats['universals_found']}\n"
            f"  HDV domains:        {len(self.hdv_system.domain_masks)}\n"
            f"  Overlap dims:       {len(self.hdv_system.find_overlaps())}\n"
            f"  Per-dim counts:     {counts}\n"
        )

    # ── Worker threads ─────────────────────────────────────────────────────────

    def _math_arxiv_thread(self):
        """
        Thread: math dimension.

        Scans ingested/*.json for arXiv URLs, downloads LaTeX source,
        extracts equations, encodes to HDV, records for discovery.
        Resumes from where it left off each scan cycle.
        """
        processed: set = set()
        storage = Path(self.ingested_dir)

        while not self._stop_event.is_set():
            paper_files = [
                f for f in storage.glob("*.json")
                if f.name != "seen_urls.json" and f.stem not in processed
            ]

            for pf in paper_files:
                if self._stop_event.is_set():
                    break

                try:
                    data = json.loads(pf.read_text())
                except Exception:
                    processed.add(pf.stem)
                    continue

                url = data.get("url", "")
                if "arxiv.org" not in url:
                    processed.add(pf.stem)
                    continue

                result = self.arxiv_parser.parse_arxiv_paper(url)
                processed.add(pf.stem)

                if not (result and result["equations"]):
                    continue

                title = data.get("article", {}).get("title", "")
                domain = self.function_library._infer_domain(url, title)

                for eq_latex in result["equations"]:
                    expr = self.eq_parser.parse(eq_latex)
                    func_type = self.eq_parser.classify_function_type(expr)
                    hdv_vec = self.hdv_system.encode_equation(eq_latex, domain)

                    self.discovery.record_pattern(
                        "math", hdv_vec,
                        {
                            "type": func_type,
                            "content": eq_latex[:100],
                            "domain": domain,
                            "paper_id": result["paper_id"],
                        },
                    )

                with self._lock:
                    self.stats["math_patterns"] += len(result["equations"])

            # Re-scan every 5 minutes for newly ingested papers
            self._stop_event.wait(300)

    def _math_library_thread(self):
        """
        Thread: keep FunctionBasisLibrary populated and HDV mapper synced.

        Runs ingest_papers_from_storage() (processes already-fetched HTML,
        which has empty equations — use arxiv_thread for real equations).
        Mainly reassigns HDV dimensions when the library grows.
        """
        while not self._stop_event.is_set():
            count_before = len(self.function_library.library)
            self.function_library.ingest_papers_from_storage(self.ingested_dir)
            count_after = len(self.function_library.library)

            if count_after > count_before:
                self.hdv_system.hdv_mapper.assign_dimensions()
                print(f"[LibThread] Library: {count_before} → {count_after}")

            self._stop_event.wait(600)  # every 10 minutes

    def _behavioral_deepwiki_thread(self, repos: List[str]):
        """
        Thread: behavioral dimension.

        Tries DeepWiki first; falls back to GitHub API.
        Encodes workflows as HDVs and records for cross-dimensional discovery.
        """
        for repo_url in repos:
            if self._stop_event.is_set():
                break

            # Try DeepWiki first (high-signal, pre-analyzed)
            capability = self.deepwiki.parse_deepwiki_summary(repo_url)

            if capability:
                hdv_vec = self.deepwiki.encode_workflow_to_hdv(
                    capability, self.hdv_system
                )
                self.discovery.record_pattern(
                    "behavioral", hdv_vec,
                    {
                        "type": "workflow",
                        "intent": capability["intent"][:80],
                        "repo": repo_url,
                        "steps": len(capability["workflow"]),
                        "source": capability.get("source", "deepwiki"),
                    },
                )
                with self._lock:
                    self.stats["behavioral_patterns"] += 1
                continue

            # Fallback: GitHub API
            cap = self.github.extract_capability_via_api(repo_url)
            if cap:
                hdv_vec = self.hdv_system.encode_workflow(
                    cap["workflow"], domain="behavioral"
                )
                self.discovery.record_pattern(
                    "behavioral", hdv_vec,
                    {
                        "type": "workflow_github_api",
                        "intent": cap["intent"][:80],
                        "repo": repo_url,
                        "steps": len(cap["workflow"]),
                        "source": "github_api",
                    },
                )
                with self._lock:
                    self.stats["behavioral_patterns"] += 1

        print("[BehavioralThread] Finished processing repos")

    def _discovery_thread(self):
        """
        Thread: scan for cross-dimensional universals every 60 seconds.

        Runs find_universals(), prints summary if new ones found,
        saves to JSON for FICUTSUpdater to log.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(60)
            if self._stop_event.is_set():
                break

            new = self.discovery.find_universals()
            if new:
                with self._lock:
                    self.stats["universals_found"] = len(self.discovery.universals)
                self.discovery.save_universals()
                print(f"\n[Discovery] {len(new)} new universal(s)! "
                      f"Total: {len(self.discovery.universals)}")
                print(self.discovery.summary())


# ── Entry point ────────────────────────────────────────────────────────────────

def run_autonomous_learning(
    repos: Optional[List[str]] = None,
    github_token: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
):
    """
    Start the autonomous learning system and run until interrupted.

    Example:
        python -c "
        from tensor.autonomous_training import run_autonomous_learning
        run_autonomous_learning()
        "
    """
    system = AutonomousLearningSystem(github_token=github_token)
    system.start(repos=repos)

    try:
        while not (stop_event and stop_event.is_set()):
            time.sleep(120)
            system.print_status()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()
