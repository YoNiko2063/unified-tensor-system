"""
FICUTS Dimension 4: Optuna Meta-Optimizer + Visualization

Tunes IntegratedHDVSystem hyperparameters to maximize universal discovery rate.

Hyperparameters searched:
  hdv_dim             : HDV space dimensionality (500 – 5000)
  n_active_per_domain : sparse mask density per domain (50 – 500)
  similarity_threshold: cosine similarity required to call a pair universal
  n_top_equations     : how many equations to encode per trial

Objective:
  Load real math patterns (from function_library.json)
  Load real behavioral patterns (from deepwiki_workflows.json or capability_maps.json)
  Encode both → run CrossDimensionalDiscovery
  Score = overlap_dims * 0.01 + universals * 10 + mean_similarity * 5

Visualizations saved to tensor/data/viz/:
  optimization_history.html     — how objective improves across trials
  param_importances.html        — which hyperparams drive discovery
  parallel_coordinates.html     — all trials, all params, all objectives
  contour_hdv_vs_active.html    — 2D landscape: hdv_dim × n_active
  hdv_pca_scatter.html          — 2D PCA of math equation HDV vectors
  overlap_heatmap.html          — domain × dimension usage heatmap
  discovery_scatter.html        — (similarity, MDL) for all pattern pairs
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_math_patterns(
    library_path: str = "tensor/data/function_library.json",
    max_patterns: int = 100,
) -> List[Dict]:
    """Load math function entries from the library."""
    p = Path(library_path)
    if not p.exists():
        return []
    lib = json.loads(p.read_text())
    entries = [
        {"symbolic_str": v["symbolic_str"], "type": v["type"],
         "domains": v.get("domains", ["general"])}
        for v in list(lib.values())[:max_patterns]
    ]
    return entries


def _load_behavioral_patterns(max_patterns: int = 50) -> List[Dict]:
    """Load behavioral patterns from DeepWiki, capability maps, or function library.

    Sources (tried in order, first non-empty wins):
      1. tensor/data/deepwiki_workflows.json  — full DeepWiki-derived workflows
      2. tensor/data/capability_maps.json     — capability map entries
      3. tensor/data/function_library.json    — synthetic behavioral patterns
         derived from equation structure (operator terms, function type, domain)
    """
    # Source 1+2: dedicated behavioral pattern files
    candidates = [
        "tensor/data/deepwiki_workflows.json",
        "tensor/data/capability_maps.json",
    ]
    for path_str in candidates:
        p = Path(path_str)
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text())
            # deepwiki_workflows format: {url: {capability: {workflow: [...]}, hdv: [...]}}
            if isinstance(raw, dict):
                patterns = []
                for url, data in list(raw.items())[:max_patterns]:
                    cap = data.get("capability", data)
                    patterns.append({
                        "workflow": cap.get("workflow", []),
                        "intent": cap.get("intent", ""),
                        "repo": url,
                    })
                if patterns:
                    return patterns
            # list format
            if isinstance(raw, list) and raw:
                return raw[:max_patterns]
        except Exception:
            continue

    # Source 3: synthesize behavioral patterns from the function library.
    # Each equation's structural metadata (type, operator_terms, domain)
    # becomes a behavioral workflow — this bridges the gap when DeepWiki
    # data isn't available yet.
    lib_path = Path("tensor/data/function_library.json")
    if lib_path.exists():
        try:
            lib = json.loads(lib_path.read_text())
            patterns = []
            for name, entry in list(lib.items())[:max_patterns]:
                func_type = entry.get("type", "unknown")
                eq_type = entry.get("equation_type", "algebraic")
                ops = entry.get("operator_terms", [])
                domains = entry.get("domains", ["general"])
                domain = domains[0] if isinstance(domains, list) else list(domains)[0]
                params = entry.get("parameters", [])

                workflow = [func_type]
                if eq_type != "algebraic":
                    workflow.append(eq_type)
                if ops:
                    workflow.extend(ops)
                if params:
                    workflow.extend([f"param_{p}" for p in params[:3]])
                workflow.append(domain)

                patterns.append({
                    "workflow": workflow,
                    "intent": f"{eq_type} {func_type}: {entry.get('symbolic_str', '')[:60]}",
                    "repo": f"library:{name}",
                })
            if patterns:
                return patterns
        except Exception:
            pass

    return []


# ── Optuna objective ───────────────────────────────────────────────────────────

def build_objective(math_patterns: List[Dict], behavioral_patterns: List[Dict]):
    """Build a closure over loaded patterns so they're not re-loaded each trial."""

    def objective(trial):
        import optuna
        from tensor.integrated_hdv import IntegratedHDVSystem
        from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery

        # Hyperparameters
        hdv_dim = trial.suggest_int("hdv_dim", 500, 5000, step=500)
        n_active = trial.suggest_int("n_active_per_domain", 50, 500, step=50)
        sim_threshold = trial.suggest_float("similarity_threshold", 0.60, 0.95)
        n_math = trial.suggest_int("n_top_equations", 20, min(len(math_patterns), 150))

        # Build fresh system for this trial
        hdv = IntegratedHDVSystem(
            hdv_dim=hdv_dim, n_modes=10, embed_dim=64,
        )
        discovery = CrossDimensionalDiscovery(
            hdv_system=hdv,
            similarity_threshold=sim_threshold,
            universals_path="/dev/null",  # don't persist trial results
        )

        # Encode math patterns
        math_eq_list = math_patterns[:n_math]
        for entry in math_eq_list:
            domain = (
                list(entry["domains"])[0]
                if entry.get("domains")
                else "general"
            )
            vec = hdv.encode_equation(entry["symbolic_str"], domain)
            discovery.record_pattern(
                "math", vec, {"type": entry["type"], "domain": domain}
            )

        # Encode behavioral patterns
        for bp in behavioral_patterns:
            steps = bp.get("workflow", [])
            if not steps:
                continue
            vec = hdv.encode_workflow(steps, "behavioral")
            discovery.record_pattern(
                "behavioral", vec,
                {"type": "workflow", "intent": bp.get("intent", "")[:60]},
            )

        # Register domains so overlaps exist
        for domain in ["ece", "biology", "physics", "finance", "general"]:
            hdv.register_domain(domain, n_active=n_active)

        # Score
        universals = discovery.find_universals()
        n_overlaps = len(hdv.find_overlaps())
        n_uni = len(universals)
        mean_sim = (
            float(np.mean([u["similarity"] for u in universals]))
            if universals else 0.0
        )

        score = n_overlaps * 0.005 + n_uni * 10.0 + mean_sim * 5.0

        # Log diagnostics
        trial.set_user_attr("n_overlaps", n_overlaps)
        trial.set_user_attr("n_universals", n_uni)
        trial.set_user_attr("mean_similarity", round(mean_sim, 4))

        return score

    return objective


# ── Study runner ──────────────────────────────────────────────────────────────

def run_optimization(
    n_trials: int = 30,
    study_name: str = "ficuts_hdv_optimization",
    storage: Optional[str] = None,
    show_progress: bool = True,
) -> "optuna.Study":
    """
    Run Optuna study to find optimal IntegratedHDVSystem configuration.

    Args:
        n_trials:      number of hyperparameter configurations to try
        study_name:    Optuna study identifier
        storage:       optional SQLite path for persistent study
                       (e.g. "sqlite:///tensor/data/optuna.db")
        show_progress: show tqdm progress bar

    Returns:
        Completed optuna.Study object.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    math_patterns = _load_math_patterns(max_patterns=150)
    behavioral_patterns = _load_behavioral_patterns(max_patterns=50)

    print(f"[Optuna] Loaded {len(math_patterns)} math patterns, "
          f"{len(behavioral_patterns)} behavioral patterns")

    if not math_patterns:
        print("[Optuna] No math patterns found. Run populate_library_from_arxiv() first.")
        return None

    storage_arg = storage or f"sqlite:///tensor/data/optuna_{study_name}.db"
    Path("tensor/data").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_arg,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective_fn = build_objective(math_patterns, behavioral_patterns)

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=show_progress,
        gc_after_trial=True,
    )

    best = study.best_trial
    print(f"\n[Optuna] Best trial #{best.number}: score={best.value:.3f}")
    print(f"  hdv_dim={best.params['hdv_dim']}")
    print(f"  n_active_per_domain={best.params['n_active_per_domain']}")
    print(f"  similarity_threshold={best.params['similarity_threshold']:.3f}")
    print(f"  n_top_equations={best.params['n_top_equations']}")
    print(f"  → {best.user_attrs.get('n_overlaps', '?')} overlap dims, "
          f"{best.user_attrs.get('n_universals', '?')} universals")

    return study


# ── Visualization ──────────────────────────────────────────────────────────────

def generate_optuna_plots(study, viz_dir: str = "tensor/data/viz") -> List[str]:
    """
    Generate standard Optuna HTML visualizations.

    Returns list of saved file paths.
    """
    import optuna.visualization as ov

    Path(viz_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    plots = {
        "optimization_history.html":  lambda: ov.plot_optimization_history(study),
        "param_importances.html":     lambda: ov.plot_param_importances(study),
        "parallel_coordinates.html":  lambda: ov.plot_parallel_coordinate(study),
        "contour_hdv_vs_active.html": lambda: ov.plot_contour(
            study, params=["hdv_dim", "n_active_per_domain"]
        ),
        "contour_sim_vs_hdv.html":    lambda: ov.plot_contour(
            study, params=["similarity_threshold", "hdv_dim"]
        ),
        "slice_plot.html":            lambda: ov.plot_slice(study),
    }

    for filename, plot_fn in plots.items():
        try:
            fig = plot_fn()
            path = str(Path(viz_dir) / filename)
            fig.write_html(path)
            saved.append(path)
            print(f"[Viz] Saved: {path}")
        except Exception as e:
            print(f"[Viz] Skipped {filename}: {e}")

    return saved


def generate_hdv_pca_scatter(
    library_path: str = "tensor/data/function_library.json",
    viz_dir: str = "tensor/data/viz",
    hdv_dim: int = 2000,
    max_points: int = 200,
) -> Optional[str]:
    """
    2D PCA scatter of equation HDV vectors, colored by function type.

    Shows whether the hash-based structural encoding produces meaningful
    clusters (exponentials together, trig together, etc.) — a direct
    window into what the HDV space is learning geometrically.
    """
    try:
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        from tensor.integrated_hdv import IntegratedHDVSystem
    except ImportError:
        print("[Viz] sklearn not available for PCA scatter — skipping")
        return None

    lib_path = Path(library_path)
    if not lib_path.exists():
        return None

    lib = json.loads(lib_path.read_text())
    if not lib:
        return None

    hdv = IntegratedHDVSystem(hdv_dim=hdv_dim, n_modes=10, embed_dim=64)

    vectors, types, labels = [], [], []
    for name, entry in list(lib.items())[:max_points]:
        domain = list(entry.get("domains", ["general"]))[0]
        vec = hdv.encode_equation(entry["symbolic_str"], domain)
        vectors.append(vec)
        types.append(entry.get("type", "unknown"))
        labels.append(f"{name}: {entry['symbolic_str'][:40]}")

    if len(vectors) < 3:
        return None

    X = np.stack(vectors)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    # Color by function type
    unique_types = sorted(set(types))
    color_map = {t: i for i, t in enumerate(unique_types)}
    colors = [color_map[t] for t in types]

    fig = go.Figure()
    for func_type in unique_types:
        mask = [i for i, t in enumerate(types) if t == func_type]
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            name=func_type,
            text=[labels[i] for i in mask],
            marker=dict(size=8, opacity=0.8),
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    var_explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=dict(
            text=f"Math Equation HDV Space — 2D PCA<br>"
                 f"<sub>PC1={var_explained[0]:.1%} var, "
                 f"PC2={var_explained[1]:.1%} var — "
                 f"{len(vectors)} equations from {len(unique_types)} types</sub>",
            font=dict(size=16),
        ),
        xaxis_title=f"PC1 ({var_explained[0]:.1%} variance)",
        yaxis_title=f"PC2 ({var_explained[1]:.1%} variance)",
        legend_title="Function Type",
        template="plotly_dark",
        height=650,
    )

    path = str(Path(viz_dir) / "hdv_pca_scatter.html")
    fig.write_html(path)
    print(f"[Viz] Saved: {path}")
    return path


def generate_overlap_heatmap(
    library_path: str = "tensor/data/function_library.json",
    viz_dir: str = "tensor/data/viz",
    hdv_dim: int = 2000,
) -> Optional[str]:
    """
    Domain × function-type heatmap of HDV overlap activity.

    Each cell = how many HDV dimensions are active for that (domain, type) pair.
    Shows which function types are most cross-domain universal.
    """
    try:
        import plotly.graph_objects as go
        from tensor.integrated_hdv import IntegratedHDVSystem
    except ImportError:
        return None

    lib_path = Path(library_path)
    if not lib_path.exists():
        return None
    lib = json.loads(lib_path.read_text())
    if not lib:
        return None

    hdv = IntegratedHDVSystem(hdv_dim=hdv_dim, n_modes=10, embed_dim=64)

    # Collect all domains and types
    all_domains: set = set()
    all_types: set = set()
    for entry in lib.values():
        all_domains.update(entry.get("domains", ["general"]))
        all_types.add(entry.get("type", "unknown"))

    all_domains_l = sorted(all_domains)
    all_types_l = sorted(all_types)

    # Build activity matrix: domain × type → active HDV dim count
    matrix = np.zeros((len(all_domains_l), len(all_types_l)), dtype=int)

    for entry in lib.values():
        func_type = entry.get("type", "unknown")
        if func_type not in all_types_l:
            continue
        type_idx = all_types_l.index(func_type)

        for domain in entry.get("domains", ["general"]):
            if domain not in all_domains_l:
                continue
            dom_idx = all_domains_l.index(domain)
            vec = hdv.encode_equation(entry["symbolic_str"], domain)
            matrix[dom_idx, type_idx] += int(vec.sum())

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_types_l,
        y=all_domains_l,
        colorscale="Viridis",
        text=matrix,
        texttemplate="%{text}",
        hovertemplate="Domain: %{y}<br>Type: %{x}<br>Active dims: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="HDV Activity: Domain × Function Type<br>"
              "<sub>Higher = more HDV dimensions active for this (domain, type) pair "
              "→ more structural overlap available for universal discovery</sub>",
        xaxis_title="Function Type",
        yaxis_title="Domain",
        template="plotly_dark",
        height=500,
    )

    path = str(Path(viz_dir) / "overlap_heatmap.html")
    fig.write_html(path)
    print(f"[Viz] Saved: {path}")
    return path


def generate_discovery_scatter(
    viz_dir: str = "tensor/data/viz",
    hdv_dim: int = 2000,
    sim_threshold: float = 0.85,
    n_math: int = 80,
    n_behavioral: int = 20,
) -> Optional[str]:
    """
    Scatter plot of (similarity, MDL) for all cross-dimensional pattern pairs.

    Each point = one (math equation, behavioral workflow) pair.
    Highlighted in gold = universals (similarity > threshold).

    This is the direct visual of the discovery mechanism:
    the math that appears to have the same structure as real-world code.
    """
    try:
        import plotly.graph_objects as go
        from tensor.integrated_hdv import IntegratedHDVSystem
        from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery
    except ImportError:
        return None

    # Load patterns
    math_patterns = _load_math_patterns(max_patterns=n_math)
    behavioral_patterns = _load_behavioral_patterns(max_patterns=n_behavioral)

    if not math_patterns or not behavioral_patterns:
        # Generate an empty chart showing 0 universals rather than returning None.
        # This keeps the visual feedback loop alive so the user sees progress.
        try:
            fig = go.Figure()
            n_m = len(math_patterns) if math_patterns else 0
            n_b = len(behavioral_patterns) if behavioral_patterns else 0
            missing = []
            if not math_patterns:
                missing.append("math (run populate_library_from_arxiv)")
            if not behavioral_patterns:
                missing.append("behavioral (run autonomous_training)")
            fig.add_annotation(
                text=f"Waiting for patterns: {', '.join(missing)}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="gray"),
            )
            fig.update_layout(
                title=dict(
                    text=f"Cross-Dimensional Discovery Landscape<br>"
                         f"<sub>{n_m} math x {n_b} behavioral = 0 pairs | "
                         f"0 universals found</sub>",
                    font=dict(size=16),
                ),
                xaxis_title="Overlap Cosine Similarity",
                yaxis_title="MDL Score",
                template="plotly_dark",
                height=600,
            )
            Path(viz_dir).mkdir(parents=True, exist_ok=True)
            path = str(Path(viz_dir) / "discovery_scatter.html")
            fig.write_html(path)
            print(f"[Viz] Saved: {path} (0 universals — waiting for patterns)")
            return path
        except Exception:
            return None

    hdv = IntegratedHDVSystem(hdv_dim=hdv_dim, n_modes=10, embed_dim=64)
    discovery = CrossDimensionalDiscovery(
        hdv_system=hdv,
        similarity_threshold=sim_threshold,
        universals_path="/dev/null",
    )

    # Encode math
    math_vecs = []
    for entry in math_patterns:
        domain = list(entry.get("domains", ["general"]))[0]
        vec = hdv.encode_equation(entry["symbolic_str"], domain)
        math_vecs.append((vec, entry))
        discovery.record_pattern("math", vec, {"type": entry["type"]})

    # Encode behavioral
    beh_vecs = []
    for bp in behavioral_patterns:
        steps = bp.get("workflow", [])
        if not steps:
            continue
        vec = hdv.encode_workflow(steps, "behavioral")
        beh_vecs.append((vec, bp))
        discovery.record_pattern("behavioral", vec, {"type": "workflow"})

    # Register domains
    for d in ["ece", "biology", "physics", "finance", "general"]:
        hdv.register_domain(d, n_active=100)

    # Compute all pair similarities + MDL
    sims, mdls, hover_texts, is_universal = [], [], [], []

    for m_vec, m_entry in math_vecs:
        for b_vec, b_entry in beh_vecs:
            sim = hdv.compute_overlap_similarity(m_vec, b_vec)
            mdl = discovery._compute_mdl(m_vec, b_vec)
            label = (
                f"Math: {m_entry['type']} — {m_entry['symbolic_str'][:40]}<br>"
                f"Code: {b_entry.get('intent', '')[:50]}<br>"
                f"Similarity: {sim:.3f} | MDL: {mdl:.3f}"
            )
            sims.append(sim)
            mdls.append(mdl)
            hover_texts.append(label)
            is_universal.append(sim >= sim_threshold)

    if not sims:
        return None

    sims_arr = np.array(sims)
    mdls_arr = np.array(mdls)
    mask_uni = np.array(is_universal)
    mask_non = ~mask_uni

    fig = go.Figure()

    # Non-universals: blue
    if mask_non.any():
        fig.add_trace(go.Scatter(
            x=sims_arr[mask_non], y=mdls_arr[mask_non],
            mode="markers",
            name="Candidate patterns",
            text=[hover_texts[i] for i in range(len(hover_texts)) if mask_non[i]],
            marker=dict(color="steelblue", size=6, opacity=0.5),
            hovertemplate="%{text}<extra></extra>",
        ))

    # Universals: gold
    if mask_uni.any():
        fig.add_trace(go.Scatter(
            x=sims_arr[mask_uni], y=mdls_arr[mask_uni],
            mode="markers",
            name=f"Universals (sim ≥ {sim_threshold})",
            text=[hover_texts[i] for i in range(len(hover_texts)) if mask_uni[i]],
            marker=dict(color="gold", size=12, symbol="star",
                        line=dict(color="white", width=1)),
            hovertemplate="%{text}<extra></extra>",
        ))

    # Threshold line
    fig.add_vline(
        x=sim_threshold, line_dash="dash", line_color="gold", opacity=0.6,
        annotation_text=f"threshold={sim_threshold}",
        annotation_position="top right",
    )

    n_found = int(mask_uni.sum())
    fig.update_layout(
        title=dict(
            text=f"Cross-Dimensional Discovery Landscape<br>"
                 f"<sub>{len(math_vecs)} math patterns × {len(beh_vecs)} behavioral patterns "
                 f"= {len(sims)} pairs | {n_found} universals found ✦</sub>",
            font=dict(size=16),
        ),
        xaxis_title="Overlap Cosine Similarity (higher = more universal)",
        yaxis_title="MDL Score (lower = better explanation)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark",
        height=600,
    )

    path = str(Path(viz_dir) / "discovery_scatter.html")
    fig.write_html(path)
    print(f"[Viz] Saved: {path} ({n_found} universals highlighted)")
    return path


def generate_function_type_distribution(
    library_path: str = "tensor/data/function_library.json",
    viz_dir: str = "tensor/data/viz",
) -> Optional[str]:
    """Bar chart of function type counts in the library."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    p = Path(library_path)
    if not p.exists():
        return None

    lib = json.loads(p.read_text())
    type_counts: Dict[str, int] = {}
    for entry in lib.values():
        t = entry.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    types_sorted = sorted(type_counts.keys(), key=lambda k: -type_counts[k])
    counts = [type_counts[t] for t in types_sorted]

    fig = go.Figure(go.Bar(
        x=types_sorted, y=counts,
        text=counts, textposition="auto",
        marker_color="steelblue",
    ))
    fig.update_layout(
        title=f"Math Function Library — Type Distribution ({sum(counts)} total functions)",
        xaxis_title="Function Type",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
    )

    path = str(Path(viz_dir) / "function_type_distribution.html")
    fig.write_html(path)
    print(f"[Viz] Saved: {path}")
    return path


def generate_all_visualizations(
    study=None,
    viz_dir: str = "tensor/data/viz",
) -> List[str]:
    """
    Generate the full visualization suite.

    Optuna plots require a completed study. Custom plots work standalone.
    """
    saved = []
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    print("\n[Viz] Generating function type distribution...")
    p = generate_function_type_distribution(viz_dir=viz_dir)
    if p:
        saved.append(p)

    print("[Viz] Generating HDV PCA scatter...")
    p = generate_hdv_pca_scatter(viz_dir=viz_dir)
    if p:
        saved.append(p)

    print("[Viz] Generating overlap heatmap...")
    p = generate_overlap_heatmap(viz_dir=viz_dir)
    if p:
        saved.append(p)

    print("[Viz] Generating discovery scatter...")
    p = generate_discovery_scatter(viz_dir=viz_dir)
    if p:
        saved.append(p)

    if study is not None and len(study.trials) >= 2:
        print("[Viz] Generating Optuna plots...")
        optuna_plots = generate_optuna_plots(study, viz_dir=viz_dir)
        saved.extend(optuna_plots)

    print(f"\n[Viz] {len(saved)} visualizations saved to {viz_dir}/")
    return saved


def open_dashboard(viz_dir: str = "tensor/data/viz"):
    """Serve visualizations via a simple HTTP server and open in browser."""
    import http.server
    import os
    import threading
    import webbrowser

    abs_viz_dir = str(Path(viz_dir).resolve())
    html_files = sorted(Path(abs_viz_dir).glob("*.html"))

    if not html_files:
        print(f"[Dashboard] No HTML files found in {viz_dir}")
        print("  Run: python run_autonomous.py --optimize  (or --dashboard after viz exist)")
        return

    PORT = 8765
    os.chdir(abs_viz_dir)

    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("", PORT), handler)

    print(f"\n[Dashboard] Serving {len(html_files)} visualizations at http://localhost:{PORT}")
    print(f"  Available plots:")
    for f in html_files:
        print(f"    http://localhost:{PORT}/{f.name}")

    # Open first file in browser
    threading.Timer(0.5, lambda: webbrowser.open(
        f"http://localhost:{PORT}/{html_files[0].name}"
    )).start()

    print("\n[Dashboard] Press Ctrl+C to stop server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        print("[Dashboard] Server stopped.")
