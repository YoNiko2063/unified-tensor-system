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

import numpy as np

from tensor.integrated_hdv import IntegratedHDVSystem
from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery
from tensor.arxiv_pdf_parser import ArxivPDFSourceParser
from tensor.function_basis import FunctionBasisLibrary, EquationParser
from tensor.deepwiki_integration import DeepWikiWorkflowParser
from tensor.github_api_fallback import GitHubAPICapabilityExtractor
from tensor.domain_registry import DomainRegistry
from tensor.growing_network import GrowingNeuralNetwork
from tensor.code_learning import RepoCodeLearner, CodeExecutor, UnknownUnknownDetector
from tensor.simulation_trainer import SimulationTrainer


_DEFAULT_REPOS = [
    "https://github.com/huggingface/transformers",
    "https://github.com/pytorch/pytorch",
    "https://github.com/numpy/numpy",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/Ultimaker/Cura",
    "https://github.com/prusa3d/PrusaSlicer",
    "https://github.com/langchain-ai/langchain",
    "https://github.com/matplotlib/matplotlib",
    # Math-adjacent repos (generate vocabulary shared with equations)
    "https://github.com/scipy/scipy",
    "https://github.com/sympy/sympy",
    "https://github.com/statsmodels/statsmodels",
    "https://github.com/tensorflow/tensorflow",
    "https://github.com/openai/gym",
    "https://github.com/networkx/networkx",
    "https://github.com/cvxpy/cvxpy",
    "https://github.com/casadi/casadi",
    "https://github.com/gekko-package/gekko",
    "https://github.com/python-control/python-control",
    "https://github.com/josephmisiti/awesome-machine-learning",
    "https://github.com/microsoft/onnxruntime",
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
        # Threshold 0.15: appropriate for token-hash overlap similarity.
        # - Random baseline cosine ≈ 0.0075 (25 active dims, 3333 universal dims)
        # - 0.15 ≈ 20× above random → requires ≥3 shared vocabulary tokens
        # - 0.85 (original) was for trained semantic embeddings; too strict for hashing
        self.discovery = CrossDimensionalDiscovery(self.hdv_system, similarity_threshold=0.15)

        # 150-domain expansion registry + self-growing network
        self.domain_registry = DomainRegistry(hdv_system=self.hdv_system)
        self.growing_net = GrowingNeuralNetwork(
            network=self.hdv_system.network,
            domain_registry=self.domain_registry,
            hdv_system=self.hdv_system,
            growth_error_threshold=0.5,
        )

        # Code learning from local repos + execution feedback loop
        self.code_learner = RepoCodeLearner(
            self.hdv_system, self.discovery, self.domain_registry
        )
        self.code_executor = CodeExecutor(timeout=5.0)
        self.unknown_detector = UnknownUnknownDetector(
            self.domain_registry, self.discovery
        )

        # Physical simulation trainer (circuit + mechanical)
        self.simulation_trainer = SimulationTrainer(
            self.hdv_system, self.discovery, self.domain_registry
        )

        # Local repos the system has direct access to (no network needed)
        self._local_repos = [
            "ecemath",     # Circuit math: Jacobian, stability, MNA, regime detection
            "dev-agent",   # Agent code: multi-modal reasoning, planning, execution
        ]

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
            "physical_patterns": 0,
            "universals_found": 0,
            "active_domains": self.domain_registry.n_active(),
            "growth_events": 0,
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
            ("domain-expand",    self._domain_expansion_thread,  ()),
            ("html-patterns",    self._html_pattern_thread,       ()),
            ("code-learning",    self._code_learning_thread,      ()),
            ("simulation",       self._simulation_thread,         ()),
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

        dom_status = self.domain_registry.status()
        grow_status = self.growing_net.status()
        print(
            f"\n[ALS Status @ {uptime_h:.2f}h]\n"
            f"  Math patterns:      {stats['math_patterns']}\n"
            f"  Behavioral patterns:{stats['behavioral_patterns']}\n"
            f"  Physical patterns:  {stats['physical_patterns']}\n"
            f"  Universals found:   {stats['universals_found']}\n"
            f"  HDV domains:        {len(self.hdv_system.domain_masks)}\n"
            f"  Overlap dims:       {len(self.hdv_system.find_overlaps())}\n"
            f"  Per-dim counts:     {counts}\n"
            f"  Domain coverage:    {dom_status['active']}/150 "
            f"({dom_status['coverage_pct']}%) active\n"
            f"  Network growth:     {grow_status['growth_events']} expansions, "
            f"worst-head err={grow_status['worst_head_error']}\n"
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

                # Classify paper into one of the 150 expansion domains
                # and activate that domain's HDV subspace + mode head.
                expanded_domain, head_idx = self.domain_registry.activate_for_text(
                    title + " " + domain, self.hdv_system
                )
                if expanded_domain != "unknown":
                    with self._lock:
                        self.stats["active_domains"] = self.domain_registry.n_active()

                # Paper title carries natural language shared with behavioral descriptions
                # e.g. "Attention Is All You Need" → tokens "attention", "need" overlap
                # with behavioral workflow steps mentioning the same concepts.
                title_vec = self.hdv_system.structural_encode(title, domain) if title else None

                for eq_latex in result["equations"]:
                    expr = self.eq_parser.parse(eq_latex)
                    func_type = self.eq_parser.classify_function_type(expr)
                    # Blend: structured (SymPy type) + raw latex tokens + paper title words
                    # This 3-way blend maximises shared vocabulary with behavioral patterns.
                    structured_vec = self.hdv_system.encode_equation(eq_latex, domain)
                    raw_vec = self.hdv_system.structural_encode(eq_latex, domain)
                    hdv_vec = np.clip(structured_vec + raw_vec, 0.0, 1.0)
                    if title_vec is not None:
                        hdv_vec = np.clip(hdv_vec + title_vec, 0.0, 1.0)

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
        Loops indefinitely — re-processes repos every 10 minutes so behavioral
        patterns keep growing alongside math patterns.
        """
        while not self._stop_event.is_set():
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

            print("[BehavioralThread] Completed one pass over repos; sleeping 600s")
            self._stop_event.wait(600)

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

    def _domain_expansion_thread(self):
        """
        Thread: 150-domain expansion monitor.

        Every 90 seconds:
          1. Checks GrowingNeuralNetwork for overloaded heads
          2. Triggers self-directed growth if threshold exceeded
          3. Logs activated domains and growth events
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(90)
            if self._stop_event.is_set():
                break

            try:
                # Self-directed grow: uses its own error history
                new_head = self.growing_net.self_directed_grow()
                if new_head is not None:
                    with self._lock:
                        self.stats["growth_events"] = len(self.growing_net.growth_history)
                        self.stats["active_domains"] = self.domain_registry.n_active()

                # Log current domain coverage periodically
                dom_status = self.domain_registry.status()
                if dom_status["active"] > 0 and dom_status["active"] % 5 == 0:
                    active_list = dom_status["active_list"][:5]
                    print(f"[DomainExpand] {dom_status['active']}/150 domains active "
                          f"({dom_status['coverage_pct']}%): {active_list}...")
            except Exception as e:
                pass  # Don't crash the thread on transient errors

    def _html_pattern_thread(self):
        """
        Thread: HTML structural pattern learning.

        Processes ingested paper HTMLs to learn which structural features
        (heading density, equation density, code block density, DOM depth)
        correlate with which domains. Encodes features as HDV vectors in
        the 'structural' domain for cross-dimensional discovery.

        This helps the system learn optimal input parameters from
        the structure of scientific papers themselves.
        """
        storage = Path(self.ingested_dir)
        processed_html: set = set()

        while not self._stop_event.is_set():
            paper_files = [
                f for f in storage.glob("*.json")
                if f.name != "seen_urls.json" and f.stem not in processed_html
            ]

            for pf in paper_files:
                if self._stop_event.is_set():
                    break

                try:
                    data = json.loads(pf.read_text())
                except Exception:
                    processed_html.add(pf.stem)
                    continue

                article = data.get("article", {})
                sections = article.get("sections", [])
                code_blocks = article.get("code_blocks", [])
                concepts = data.get("concepts", {})

                if not sections:
                    processed_html.add(pf.stem)
                    continue

                # Extract structural features from HTML
                heading_count = sum(1 for s in sections if s.get("level", 99) <= 3)
                text_sections = sum(1 for s in sections if s.get("level", 0) > 3)
                eq_count = len(concepts.get("equations", []))
                term_count = len(concepts.get("technical_terms", []))
                dom_depth = article.get("dom_depth", 0)
                code_count = len(code_blocks)

                # Build structural descriptor text for HDV encoding
                # (dense natural language → token overlap with other domains)
                total_sections = max(len(sections), 1)
                structural_desc = (
                    f"heading sections {heading_count} "
                    f"equation density {eq_count} "
                    f"technical terms {term_count} "
                    f"code blocks {code_count} "
                    f"dom depth {dom_depth} "
                    f"text sections {text_sections}"
                )

                # Also classify which of the 150 domains this paper belongs to
                title = article.get("title", "")
                tech_terms_str = " ".join(
                    t for t in concepts.get("technical_terms", [])[:10]
                )
                classification_text = f"{title} {tech_terms_str}"
                domain_id = self.domain_registry.best_domain(classification_text)
                if domain_id != "unknown":
                    self.domain_registry.activate_domain(domain_id, self.hdv_system)

                # Encode structural pattern
                hdv_vec = self.hdv_system.structural_encode(structural_desc, "structural")
                self.discovery.record_pattern(
                    "math", hdv_vec,
                    {
                        "type": "html_structural",
                        "heading_count": heading_count,
                        "eq_count": eq_count,
                        "code_count": code_count,
                        "domain": domain_id,
                        "paper": pf.stem[:16],
                    },
                )

                processed_html.add(pf.stem)

            # Re-scan every 5 minutes for new ingested papers
            self._stop_event.wait(300)

    def _simulation_thread(self):
        """
        Thread: physical dimension.

        Runs circuit and mechanical simulations via ecemath, encodes the
        eigenvalue structure and stability information as HDV vectors in
        the 'physical' dimension.

        This populates the physical dimension so that CrossDimensionalDiscovery
        can find universals between:
          - 'physical' (eigenvalues from RC/RLC/spring-mass simulations)
          - 'math' (arXiv papers on stability, eigenvalues, ODE systems)
          - 'execution' (ecemath code: numerical_jacobian, stability_analysis)

        PoC cross-domain proof: RLC ↔ spring-mass → same operator algebra
        → shared Pontryagin characters → overlap similarity > threshold
        (verifies LOGIC_FLOW.md Section 9 Phase 9).
        """
        # Initial pass: small sweep to bootstrap physical patterns quickly
        try:
            result = self.simulation_trainer.run_one_pass(n_sweep=3)
            self.simulation_trainer.print_report(result)
            with self._lock:
                self.stats["physical_patterns"] = result["total_encoded"]
        except Exception as e:
            print(f"[SimThread] Initial pass failed: {e}")

        # Extended sweep: more parameter points for richer coverage
        while not self._stop_event.is_set():
            self._stop_event.wait(600)  # every 10 minutes
            if self._stop_event.is_set():
                break
            try:
                result = self.simulation_trainer.run_one_pass(n_sweep=4)
                self.simulation_trainer.print_report(result)
                with self._lock:
                    self.stats["physical_patterns"] = result["total_encoded"]
            except Exception as e:
                print(f"[SimThread] Extended pass failed: {e}")

    def _code_learning_thread(self):
        """
        Thread: learn from local git repositories (ecemath, dev-agent) and
        from repos discovered by the unknown-unknown detector.

        Workflow:
          1. Learn from local repos on startup (ecemath: circuit math,
             dev-agent: agent/reasoning code)
          2. Periodically check for 'unknown unknown' domains (gaps in coverage)
          3. For each gap domain, recommend and ingest target repos
          4. Run code snippets from ecemath to generate execution feedback

        This creates the self-reinforcing loop:
          local code → HDV patterns → universals discovered → grow network →
          detect gaps → find repos → ingest → more patterns → ...
        """
        # Phase 1: learn from local repos immediately
        for repo_path in self._local_repos:
            if self._stop_event.is_set():
                return
            n = self.code_learner.learn_from_repo(repo_path, max_files=300)
            if n > 0:
                print(f"[CodeThread] Learned {n} functions from '{repo_path}'")
            with self._lock:
                self.stats["active_domains"] = self.domain_registry.n_active()

        # Phase 2: execute a few ecemath examples to generate execution patterns
        _ECEMATH_SAMPLES = [
            # Simple circuits that can run without ecemath import
            "import numpy as np; J = np.array([[0, -1], [1, -2.0]]); print(np.linalg.eigvals(J))",
            "import numpy as np; A = np.eye(3) * -1; print('stable:', all(np.real(np.linalg.eigvals(A)) < 0))",
            "import numpy as np; x = np.linspace(0, 2*np.pi, 10); print('sin sum:', round(float(np.sum(np.sin(x))), 3))",
        ]
        for code_snippet in _ECEMATH_SAMPLES:
            if self._stop_event.is_set():
                return
            hdv_vec = self.code_executor.execute_and_encode(code_snippet, self.hdv_system)
            if hdv_vec is not None:
                result = self.code_executor.execute(code_snippet)
                self.discovery.record_pattern(
                    "execution", hdv_vec,
                    {
                        "type": "executed_snippet",
                        "success": result["success"],
                        "output_type": result["output_type"],
                        "code_preview": code_snippet[:60],
                        "elapsed_ms": int(result["elapsed"] * 1000),
                    },
                )

        # Phase 3: periodic gap analysis + new repo discovery
        while not self._stop_event.is_set():
            self._stop_event.wait(300)  # every 5 minutes
            if self._stop_event.is_set():
                break

            # Report coverage gaps (unknown unknowns)
            report = self.unknown_detector.coverage_report()
            covered = report["covered"]
            total = report["total_domains"]
            if covered < total:
                # Find the most important uncovered domain and recommend repos
                top_uncovered = report["top_uncovered"][:3]
                for domain_id in top_uncovered:
                    suggested = self.unknown_detector.recommend_repos_for_domain(domain_id)
                    for repo_url in suggested:
                        # Ingest the suggested repo via GitHub API (fallback)
                        try:
                            cap = self.github.extract_capability_via_api(repo_url)
                            if cap:
                                hdv_vec = self.hdv_system.encode_workflow(
                                    cap["workflow"], domain="execution"
                                )
                                self.discovery.record_pattern(
                                    "execution", hdv_vec,
                                    {
                                        "type": "gap_fill_repo",
                                        "gap_domain": domain_id,
                                        "repo": repo_url,
                                        "intent": cap["intent"][:60],
                                    },
                                )
                                # Activate the gap domain
                                self.domain_registry.activate_domain(
                                    domain_id, self.hdv_system
                                )
                        except Exception:
                            pass

            with self._lock:
                self.stats["active_domains"] = self.domain_registry.n_active()


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
