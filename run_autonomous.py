#!/usr/bin/env python3
"""
FICUTS Autonomous Learning System — Unified Entry Point

Usage:
  python run_autonomous.py                   Start full autonomous system
  python run_autonomous.py --populate        Populate function library from arXiv
  python run_autonomous.py --curriculum      Train on freeCodeCamp / books / geometry
  python run_autonomous.py --discover        Find repos that fill capability gaps
  python run_autonomous.py --bootstrap       Attempt Scrapling/Open3D/PrusaSlicer/SecretKnowledge integration
  python run_autonomous.py --predict TEXT    Run prediction-driven learning loop on TEXT file
  python run_autonomous.py --deq TEXT TYPE   Solve TEXT as a DEQ of TYPE (paper|circuit|code|3d_model)
  python run_autonomous.py --optimize        Run Optuna meta-optimization
  python run_autonomous.py --visualize       Generate all visualization HTML files
  python run_autonomous.py --dashboard       Generate viz + serve in browser
  python run_autonomous.py --status          Show current system status

Combined examples:
  python run_autonomous.py --populate --curriculum --discover
  python run_autonomous.py --bootstrap --optimize --visualize
  python run_autonomous.py --populate --curriculum --discover --optimize --trials 30

All data stored in tensor/data/:
  function_library.json     math equations from arXiv papers
  deepwiki_workflows.json   behavioral patterns from DeepWiki + GitHub
  universals.json           discovered cross-dimensional patterns
  hdv_state.json            HDV space domain masks
  optuna_*.db               Optuna study database
  viz/                      HTML visualization files
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List


# ── Status display ─────────────────────────────────────────────────────────────

def print_status():
    """Print current state of all data stores."""
    print("\n" + "=" * 60)
    print("  FICUTS System Status")
    print("=" * 60)

    # Function library
    lib_path = Path("tensor/data/function_library.json")
    if lib_path.exists():
        lib = json.loads(lib_path.read_text())
        type_counts: dict = {}
        for v in lib.values():
            t = v.get("type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"\n  Math Dimension (function_library.json):")
        print(f"    {len(lib)} total functions")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:6]:
            bar = "█" * min(c, 40)
            print(f"    {t:15s} {bar} {c}")
    else:
        print(f"\n  Math Dimension: EMPTY  (run --populate)")

    # Ingested papers
    ingested_dir = Path("tensor/data/ingested")
    if ingested_dir.exists():
        n_papers = len(list(ingested_dir.glob("*.json")))
        print(f"\n  Ingested papers: {n_papers}")

    # Behavioral patterns
    beh_path = Path("tensor/data/deepwiki_workflows.json")
    cap_path = Path("tensor/data/capability_maps.json")
    for p, label in [(beh_path, "DeepWiki"), (cap_path, "Capability maps")]:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                n = len(data) if isinstance(data, (dict, list)) else 0
                print(f"\n  Behavioral Dimension ({label}): {n} entries")
            except Exception:
                pass

    # Universals
    uni_path = Path("tensor/data/universals.json")
    if uni_path.exists():
        try:
            unis = json.loads(uni_path.read_text())
            print(f"\n  Universals Discovered: {len(unis)}")
            for u in unis[:3]:
                dims = " ↔ ".join(u.get("dimensions", []))
                sim = u.get("similarity", 0)
                types = [p.get("type", "?") for p in u.get("patterns", [])]
                print(f"    [{dims}] sim={sim:.3f} | {' ≈ '.join(types)}")
        except Exception:
            print("\n  Universals: (none yet)")
    else:
        print(f"\n  Universals Discovered: 0")

    # Domain coverage (150-domain expansion system)
    dom_path = Path("tensor/data/active_domains.json")
    if dom_path.exists():
        try:
            dom_data = json.loads(dom_path.read_text())
            active = dom_data.get("active", [])
            pct = round(100.0 * len(active) / 150, 1)
            print(f"\n  Domain Coverage (150-domain expansion):")
            print(f"    Active: {len(active)}/150 ({pct}%)")
            if active:
                print(f"    Domains: {', '.join(active[:6])}{'...' if len(active) > 6 else ''}")
        except Exception:
            pass

    # HDV state
    hdv_path = Path("tensor/data/hdv_state.json")
    if hdv_path.exists():
        try:
            state = json.loads(hdv_path.read_text())
            n_domains = len(state.get("domain_masks", {}))
            usage = state.get("dim_usage", [])
            n_overlaps = sum(1 for v in usage if v >= 2)
            print(f"\n  HDV Space:")
            print(f"    Dimensions:   {state.get('hdv_dim', '?')}")
            print(f"    Domains:      {n_domains}")
            print(f"    Overlap dims: {n_overlaps}")
        except Exception:
            pass

    # Optuna
    opt_dbs = list(Path("tensor/data").glob("optuna_*.db"))
    if opt_dbs:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            for db in opt_dbs:
                name = db.stem.replace("optuna_", "")
                study = optuna.load_study(
                    study_name=name,
                    storage=f"sqlite:///{db}",
                )
                best = study.best_value if study.trials else None
                n_t = len(study.trials)
                print(f"\n  Optuna ({name}): {n_t} trials, "
                      f"best={best:.3f}" if best else f"  Optuna ({name}): {n_t} trials")
        except Exception:
            pass

    # Visualizations
    viz_dir = Path("tensor/data/viz")
    if viz_dir.exists():
        html_files = list(viz_dir.glob("*.html"))
        if html_files:
            print(f"\n  Visualizations ({len(html_files)} files in tensor/data/viz/):")
            for f in sorted(html_files):
                print(f"    {f.name}")

    print("\n" + "=" * 60 + "\n")


# ── Predict ───────────────────────────────────────────────────────────────────

def do_predict(text: str, max_iterations: int = 200):
    """Run prediction-driven learning loop on text."""
    from tensor.integrated_hdv import IntegratedHDVSystem
    from tensor.prediction_learning import ContinuousLearningLoop

    print("\n[Predict] Building HDV system...")
    hdv = IntegratedHDVSystem(hdv_dim=10000, n_modes=150, embed_dim=512)

    loop = ContinuousLearningLoop(hdv_system=hdv, max_iterations=max_iterations)
    summary = loop.run(text, verbose=True)

    print(f"\n[Predict] Done:")
    print(f"  Concepts learned : {summary['concepts_learned']}/{summary['total_concepts']}")
    print(f"  Iterations       : {summary['iterations']}")
    print(f"  Lyapunov stable  : {summary['lyapunov_stable']}")
    if summary["energy_history"]:
        final_e = summary["energy_history"][-1]
        print(f"  Final energy     : {final_e:.4f}")
    return summary


# ── DEQ Solve ─────────────────────────────────────────────────────────────────

def do_deq(input_text: str, input_type: str):
    """Convert input to DEQ and solve in HDV space."""
    from tensor.integrated_hdv import IntegratedHDVSystem
    from tensor.deq_system import UnifiedDEQSolver

    print(f"\n[DEQ] Solving {input_type} input...")
    hdv = IntegratedHDVSystem(hdv_dim=10000, n_modes=150, embed_dim=512)

    solver = UnifiedDEQSolver(hdv_system=hdv)
    result = solver.solve(input_text, input_type)

    print(f"\n[DEQ] Result:")
    print(f"  Equation   : {result['equation'][:120]}")
    print(f"  Domain     : {result['domain']}")
    print(f"  Variables  : {result['variables']}")
    print(f"  Parameters : {dict(list(result['parameters'].items())[:5])}")
    print(f"  Verified   : {result['verified']}")
    print(f"  Confidence : {result['confidence']:.4f}")
    print(f"  Similar    : {result['similar_found']} previously-solved DEQs")
    return result


# ── Discover ──────────────────────────────────────────────────────────────────

def do_discover(github_token: str = None):
    """Run capability discovery: find repos that fill current gaps."""
    from tensor.integrated_hdv import IntegratedHDVSystem
    from tensor.curriculum_trainer import CurriculumTrainer

    print("\n[Discover] Building HDV system and running capability discovery...")
    hdv = IntegratedHDVSystem(hdv_dim=10000, n_modes=150, embed_dim=512)

    trainer = CurriculumTrainer(
        hdv_system=hdv,
        rate_limit_seconds=1.5,
    )
    gaps = trainer.discover_new_capabilities()
    print(f"\n[Discover] Found {len(gaps)} capability gaps with candidate repos.")
    trainer.save_patterns()
    return gaps


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def do_bootstrap(github_token: str = None):
    """Attempt autonomous integration of 4 external resources."""
    from tensor.integrated_hdv import IntegratedHDVSystem
    from tensor.bootstrap_manager import BootstrapManager

    print("\n[Bootstrap] Building HDV system for bootstrap attempt...")
    hdv = IntegratedHDVSystem(hdv_dim=10000, n_modes=150, embed_dim=512)

    bootstrap = BootstrapManager(
        hdv_system=hdv,
        rate_limit_seconds=1.0,
        github_token=github_token,
    )
    results = bootstrap.run_bootstrap()
    return results


# ── Curriculum ────────────────────────────────────────────────────────────────

def do_curriculum(github_token: str = None):
    """Run full curriculum training (challenges + books + geometry + architecture)."""
    from tensor.integrated_hdv import IntegratedHDVSystem
    from tensor.curriculum_trainer import CurriculumTrainer

    print("\n[Curriculum] Starting curriculum training...")
    hdv = IntegratedHDVSystem(hdv_dim=10000, n_modes=150, embed_dim=512)

    trainer = CurriculumTrainer(
        hdv_system=hdv,
        rate_limit_seconds=1.5,
    )
    n = trainer.train()
    print(f"\n[Curriculum] Done. Encoded {n} total patterns.")
    return trainer


# ── Populate ──────────────────────────────────────────────────────────────────

def do_populate(max_papers: int = None):
    """Download LaTeX source from arXiv, extract equations, populate library."""
    from tensor.function_basis import populate_library_from_arxiv

    print("\n[Populate] Starting function library population from arXiv...")
    print(f"[Populate] This downloads LaTeX source for up to "
          f"{'all' if max_papers is None else max_papers} papers.")
    print("[Populate] Rate-limited to ~1 request/1.5s — this may take a while.\n")

    lib = populate_library_from_arxiv(max_papers=max_papers)
    print(f"\n[Populate] Done. Library has {len(lib.library)} functions.")
    return lib


# ── Optimize ──────────────────────────────────────────────────────────────────

def do_optimize(n_trials: int = 30):
    """Run Optuna meta-optimization and return the completed study."""
    from tensor.meta_optimizer import run_optimization

    print(f"\n[Optimize] Running {n_trials} Optuna trials...")
    print("[Optimize] Each trial builds a fresh IntegratedHDVSystem, encodes patterns,")
    print("[Optimize] and measures universal discovery rate.\n")

    study = run_optimization(n_trials=n_trials, show_progress=True)

    if study is None:
        print("[Optimize] Aborted — no math patterns available. Run --populate first.")
        return None

    return study


# ── Visualize ─────────────────────────────────────────────────────────────────

def do_visualize(study=None):
    """Generate all HTML visualization files."""
    from tensor.meta_optimizer import generate_all_visualizations

    print("\n[Visualize] Generating visualization suite...")
    saved = generate_all_visualizations(study=study, viz_dir="tensor/data/viz")

    if not saved:
        print("[Visualize] No files generated. Ensure math patterns are loaded.")
    else:
        print(f"\n[Visualize] {len(saved)} files ready:")
        for path in saved:
            print(f"    {path}")
        print("\n  Open any file in your browser, or run --dashboard to serve them.")
    return saved


# ── Dashboard ─────────────────────────────────────────────────────────────────

def do_dashboard():
    """Generate visualizations and serve them via HTTP."""
    from tensor.meta_optimizer import open_dashboard

    # Generate first if needed
    viz_dir = Path("tensor/data/viz")
    if not list(viz_dir.glob("*.html")):
        print("[Dashboard] No visualizations found — generating first...")
        do_visualize()

    open_dashboard()  # blocks until Ctrl+C


# ── Live Dashboard ─────────────────────────────────────────────────────────────

def do_dashboard_live(port: int = 8766, refresh_seconds: int = 5):
    """
    Start a live-updating dashboard.

    Serves tensor/data/viz/ via HTTP; injects auto-refresh every
    `refresh_seconds` seconds so Optuna plots update as trials complete.
    Blocks until Ctrl+C.
    """
    import http.server
    import socketserver
    import threading

    viz_dir = Path("tensor/data/viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Generate initial snapshot
    do_visualize()

    REFRESH_JS = (
        f'<script>setTimeout(()=>location.reload(),{refresh_seconds * 1000})</script>'
    ).encode()

    class LiveHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viz_dir), **kwargs)

        def do_GET(self):
            # Serve index listing auto-refreshed
            if self.path in ("/", ""):
                self.path = "/"
                files = sorted(viz_dir.glob("*.html"))
                body = "<html><head></head><body><h2>FICUTS Live Dashboard</h2><ul>"
                for f in files:
                    body += f'<li><a href="/{f.name}">{f.name}</a></li>'
                body += "</ul>" + REFRESH_JS.decode() + "</body></html>"
                encoded = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            super().do_GET()

        def log_message(self, *_):
            pass  # suppress per-request noise

    def _regen_loop():
        """Regenerate visualizations every refresh_seconds."""
        while True:
            time.sleep(refresh_seconds)
            try:
                do_visualize()
            except Exception:
                pass

    regen = threading.Thread(target=_regen_loop, daemon=True)
    regen.start()

    with socketserver.TCPServer(("", port), LiveHandler) as httpd:
        print(f"\n[Dashboard-Live] Serving at http://localhost:{port}/")
        print(f"[Dashboard-Live] Auto-refreshes every {refresh_seconds}s. Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Dashboard-Live] Stopped.")


# ── Parallel Ingest ────────────────────────────────────────────────────────────

def do_ingest(paper_ids: List[str], num_workers: int = 4):
    """Ingest arXiv papers in parallel using ParallelPaperIngester."""
    from tensor.integrated_hdv import IntegratedHDVSystem
    from tensor.parallel_ingestion import ParallelPaperIngester

    print(f"\n[Ingest] Building HDV system...")
    hdv = IntegratedHDVSystem(hdv_dim=10000, n_modes=150, embed_dim=512)

    ingester = ParallelPaperIngester(
        hdv_system=hdv, num_workers=num_workers, rate_limit_seconds=1.5
    )

    print(f"[Ingest] Processing {len(paper_ids)} papers with {num_workers} workers...")
    t0 = time.time()
    results = ingester.ingest_batch(paper_ids)
    elapsed = time.time() - t0

    s = ingester.summary()
    print(f"\n[Ingest] Done in {elapsed:.1f}s:")
    print(f"  Papers processed  : {s['papers_processed']}")
    print(f"  Equations ingested: {s['equations_ingested']}")
    print(f"  Final energy      : {s['final_energy']:.4f}")
    print(f"  Lyapunov stable   : {s['lyapunov_stable']}")
    if elapsed > 0:
        print(f"  Throughput        : {len(results)/elapsed*60:.1f} papers/min")
    return ingester


# ── Run system ────────────────────────────────────────────────────────────────

def do_run(
    repos=None,
    github_token=None,
    monitor_interval: int = 120,
):
    """Start the full autonomous learning system."""
    from tensor.autonomous_training import AutonomousLearningSystem

    print("\n" + "=" * 60)
    print("  FICUTS Autonomous Learning System")
    print("=" * 60)
    print_status()

    system = AutonomousLearningSystem(
        hdv_dim=10000,
        n_modes=150,
        embed_dim=512,
        github_token=github_token,
    )

    system.start(repos=repos)

    print(f"[System] Running. Status every {monitor_interval}s. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(monitor_interval)
            system.print_status()

            # Auto-generate visualizations every 10 minutes
            vis_interval = 600
            if int(time.time()) % vis_interval < monitor_interval:
                print("[System] Auto-generating visualizations...")
                from tensor.meta_optimizer import generate_all_visualizations
                generate_all_visualizations(viz_dir="tensor/data/viz")

    except KeyboardInterrupt:
        print("\n[System] Interrupt received.")
    finally:
        system.stop()
        print("[System] Generating final visualizations...")
        from tensor.meta_optimizer import generate_all_visualizations
        generate_all_visualizations(viz_dir="tensor/data/viz")
        print("[System] Goodbye.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FICUTS Autonomous Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--populate", action="store_true",
        help="Populate function library from arXiv papers (rate-limited)",
    )
    parser.add_argument(
        "--max-papers", type=int, default=None,
        metavar="N",
        help="Limit papers to process during --populate (default: all 359)",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run Optuna meta-optimization to find best HDV hyperparameters",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of Optuna trials (default: 30)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate all HTML visualizations in tensor/data/viz/",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Generate visualizations and serve at http://localhost:8765",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current system status and exit",
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Run curriculum training (challenges + books + geometry + architecture)",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Discover repos that fill current capability gaps",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Attempt autonomous integration of 4 external resources (Scrapling, Open3D, PrusaSlicer, SecretKnowledge)",
    )
    parser.add_argument(
        "--predict", nargs="?", const="-", default=None,
        metavar="FILE",
        help="Run prediction-driven learning on FILE (or stdin if no file given)",
    )
    parser.add_argument(
        "--deq", nargs=2, metavar=("FILE", "TYPE"),
        help="Solve FILE as a DEQ of TYPE (paper|circuit|code|3d_model)",
    )
    parser.add_argument(
        "--ingest", nargs="+", metavar="PAPER_ID",
        help="Ingest arXiv papers in parallel (e.g. --ingest 2301.00001 2301.00002)",
    )
    parser.add_argument(
        "--ingest-workers", type=int, default=4,
        metavar="N",
        help="Number of parallel workers for --ingest (default: 4)",
    )
    parser.add_argument(
        "--dashboard-live", action="store_true",
        help="Start live auto-refreshing dashboard at http://localhost:8766",
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=8766,
        metavar="PORT",
        help="Port for --dashboard-live (default: 8766)",
    )
    parser.add_argument(
        "--github-token", type=str, default=None,
        metavar="TOKEN",
        help="GitHub personal access token (increases API rate limit)",
    )
    parser.add_argument(
        "--monitor-interval", type=int, default=120,
        metavar="SECONDS",
        help="Status print interval when running (default: 120)",
    )

    args = parser.parse_args()

    # Change to project root (so relative paths work regardless of CWD)
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Status only
    if args.status:
        print_status()
        return

    # No flags → run the full system
    run_requested = not any([
        args.populate, args.optimize, args.visualize, args.dashboard,
        args.curriculum, args.discover, args.bootstrap,
        args.predict, args.deq, args.ingest, args.dashboard_live,
    ])

    study = None
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")

    # Step 1: Populate (if requested)
    if args.populate:
        do_populate(max_papers=args.max_papers)

    # Step 2: Curriculum training (if requested)
    if args.curriculum:
        do_curriculum(github_token=github_token)

    # Step 3: Capability discovery (if requested)
    if args.discover:
        do_discover(github_token=github_token)

    # Step 4: Bootstrap external resources (if requested)
    if args.bootstrap:
        do_bootstrap(github_token=github_token)

    # Step 4b: Prediction-driven learning (if requested)
    if args.predict is not None:
        if args.predict == "-":
            text = sys.stdin.read()
        else:
            text_path = Path(args.predict)
            text = text_path.read_text() if text_path.exists() else args.predict
        do_predict(text)

    # Step 4c: DEQ solve (if requested)
    if args.deq:
        file_or_text, input_type = args.deq
        text_path = Path(file_or_text)
        text = text_path.read_text() if text_path.exists() else file_or_text
        do_deq(text, input_type)

    # Step 5: Optimize (if requested)
    if args.optimize:
        study = do_optimize(n_trials=args.trials)

    # Step 6: Visualize (if requested, or after optimize)
    if args.visualize or (args.optimize and study is not None):
        do_visualize(study=study)

    # Step 7: Dashboard (if requested)
    if args.dashboard:
        do_dashboard()
        return  # blocks until Ctrl+C

    # Step 7b: Live dashboard (if requested — blocks, must be last)
    if args.dashboard_live:
        do_dashboard_live(port=args.dashboard_port)
        return  # blocks until Ctrl+C

    # Step 7c: Parallel ingest (if requested)
    if args.ingest:
        do_ingest(args.ingest, num_workers=args.ingest_workers)

    # Step 8: Run full system (default or explicit)
    if run_requested:
        do_run(github_token=github_token, monitor_interval=args.monitor_interval)


if __name__ == "__main__":
    main()
