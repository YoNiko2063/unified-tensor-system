#!/usr/bin/env python
"""Run the configuration space explorer.

Usage:
  python run_explorer.py --target bandpass --freq 1000 --Q 10 --batch 256
  python run_explorer.py --target snn --neurons 32 --sparsity 0.8 --duration 60
  python run_explorer.py --target custom --steps 5000 --resume
  python run_explorer.py --target bandpass --duration 300  # run 5 minutes
  python run_explorer.py --gate NAND --precompute 50000 --batch 512 --duration 300
  python run_explorer.py --gate ALL --duration 60  # run all 5 gates sequentially
  python run_explorer.py --diagnose --target bandpass --freq 1000 --Q 10
  python run_explorer.py --target cross_level --duration 120
"""
import os
import sys
import argparse
import time
from multiprocessing import cpu_count

# Set BLAS threading before numpy import
n_cores = cpu_count() or 4
blas_threads = str(max(1, n_cores - 2))
os.environ.setdefault('OMP_NUM_THREADS', blas_threads)
os.environ.setdefault('OPENBLAS_NUM_THREADS', blas_threads)
os.environ.setdefault('MKL_NUM_THREADS', blas_threads)

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))
sys.path.insert(0, _ROOT)

from tensor.explorer import ConfigurationExplorer, ExplorerConfig, ExplorationTarget


def _print_report(stats, explorer, config):
    """Print final exploration report."""
    print(f"\n{'='*60}")
    print(f"  EXPLORATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Steps: {stats['steps']}")
    print(f"  Configs scored: {stats['total_configs_scored']:,}")
    print(f"  Elapsed: {stats['elapsed_seconds']:.1f}s")
    print(f"  Rate: {stats['configs_per_second']:.0f} configs/s")
    print(f"  Best score: {stats['best_score']:.4f}")
    print(f"  Results stored: {stats['results_stored']:,}")
    print(f"  Manifold RAM: {stats['manifold_ram_mb']:.0f}MB")
    print(f"  Results RAM: {stats['results_ram_mb']:.0f}MB")
    print()

    best = explorer.best_configurations(10)
    print(f"  TOP 10 CONFIGURATIONS:")
    print(f"  {'Rank':<6}{'Score':<10}{'Config':<10}{'Eigenvalues (top 4)'}")
    print(f"  {'-'*50}")
    for item in best:
        ev_str = ', '.join(f'{e:.3f}' for e in item['eigenvalues'][:4])
        print(f"  {item['rank']:<6}{item['score']:<10.4f}"
              f"{item['config_idx']:<10}[{ev_str}]")

    print(f"\n  Checkpoint saved to {config.log_dir}/explorer_checkpoint.npz")


def _run_single(args, score_fn, target_desc, log_dir=None):
    """Run a single exploration target."""
    config = ExplorerConfig(
        n_precompute=args.precompute,
        batch_size=args.batch,
        ram_limit_gb=args.ram_limit,
        target='custom',
        score_fn=score_fn,
        duration=float(args.duration) if args.duration is not None else None,
        log_dir=log_dir or 'tensor/logs',
        scale_ram=(args.ram_limit > 1.0 and not args.diagnose),
    )

    explorer = ConfigurationExplorer(config)

    # Resume or precompute
    if args.resume and explorer.load_checkpoint():
        print(f"Resumed from checkpoint: {explorer.results.count} results loaded")
    else:
        explorer.precompute(progress=True)

    # Diagnose mode: analyze and exit
    if args.diagnose:
        diag = explorer.diagnose(n_samples=min(1000, explorer.manifold.n_configs))
        print(f"\n  Mean score: {diag['mean_score']:.4f}")
        print(f"  Max score: {diag['max_score']:.4f}")
        print(f"  Std score: {diag['std_score']:.4f}")
        return None, explorer, config

    print()

    # Run: duration-based or step-based
    if args.duration is not None:
        stats = explorer.run_forever(duration=float(args.duration), progress=True)
    else:
        stats = explorer.run(args.steps, progress=True)

    _print_report(stats, explorer, config)
    return stats, explorer, config


def main():
    parser = argparse.ArgumentParser(description='Configuration Space Explorer')
    parser.add_argument('--target', default='bandpass',
                        help='Optimization target (bandpass, snn, custom, cross_level)')
    parser.add_argument('--batch', type=int, default=256,
                        help='Batch size per step')
    parser.add_argument('--ram-limit', type=float, default=40.0,
                        help='Max RAM usage in GB')
    parser.add_argument('--precompute', type=int, default=10000,
                        help='Initial grid size')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Exploration steps (ignored if --duration set)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')

    # Bandpass target parameters
    parser.add_argument('--freq', type=float, default=1000.0,
                        help='Center frequency for bandpass target (Hz)')
    parser.add_argument('--Q', type=float, default=10.0,
                        help='Q factor for bandpass target')

    # SNN target parameters
    parser.add_argument('--neurons', type=int, default=16,
                        help='Neuron count for snn target')
    parser.add_argument('--sparsity', type=float, default=0.8,
                        help='Connection sparsity for snn target (0-1)')

    # Duration-based run
    parser.add_argument('--duration', type=int, default=None,
                        help='Run for N seconds then stop (default: use --steps)')

    # Logic gate targets
    parser.add_argument('--gate', type=str, default=None,
                        choices=['NOT', 'AND', 'OR', 'NAND', 'XOR', 'ALL'],
                        help='Logic gate target (overrides --target)')

    # Diagnostic mode
    parser.add_argument('--diagnose', action='store_true',
                        help='Run diagnostic analysis (1000 samples) then exit')

    args = parser.parse_args()

    # System info
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = 48.0

    print(f"System: {n_cores} cores, {ram_gb:.0f}GB RAM")
    print(f"BLAS threads: {blas_threads}")

    # Handle --gate (overrides --target)
    if args.gate:
        if args.gate == 'ALL':
            gates = ['NOT', 'AND', 'OR', 'NAND', 'XOR']
            all_results = []
            for gate_name in gates:
                print(f"\n{'='*60}")
                print(f"  GATE: {gate_name}")
                print(f"{'='*60}")
                score_fn = ExplorationTarget.logic_gate(gate_name)
                target_desc = f"logic_gate ({gate_name})"
                print(f"Target: {target_desc}")
                print(f"Batch: {args.batch}, Precompute: {args.precompute}")
                if args.duration is not None:
                    print(f"Duration: {args.duration}s")
                else:
                    print(f"Steps: {args.steps}")

                stats, explorer, config = _run_single(
                    args, score_fn, target_desc,
                    log_dir=f'tensor/logs/gate_{gate_name}')
                if stats:
                    all_results.append((gate_name, stats))

            # Comparison table
            if all_results:
                print(f"\n{'='*60}")
                print(f"  GATE COMPARISON TABLE")
                print(f"{'='*60}")
                print(f"  {'Gate':<8}{'Best':<10}{'Rate':<12}{'Above 0.5':<12}{'Scored'}")
                print(f"  {'-'*52}")
                for gate_name, st in all_results:
                    scores = st.get('results_stored', 0)
                    print(f"  {gate_name:<8}{st['best_score']:<10.4f}"
                          f"{st['configs_per_second']:<12.0f}"
                          f"{'n/a':<12}{st['total_configs_scored']:,}")
            return
        else:
            score_fn = ExplorationTarget.logic_gate(args.gate)
            target_desc = f"logic_gate ({args.gate})"
    elif args.target == 'cross_level':
        score_fn = ExplorationTarget.cross_level_resonance(
            level_a=0, level_b=2, target_resonance=0.8)
        target_desc = "cross_level (L0 market <-> L2 code)"
    elif args.target == 'bandpass':
        score_fn = ExplorationTarget.bandpass_filter(freq=args.freq, Q=args.Q)
        target_desc = f"bandpass (freq={args.freq}Hz, Q={args.Q})"
    elif args.target == 'snn':
        score_fn = ExplorationTarget.snn_configuration(
            neurons=args.neurons, sparsity=args.sparsity)
        target_desc = f"snn (neurons={args.neurons}, sparsity={args.sparsity})"
    else:
        score_fn = None
        target_desc = args.target

    print(f"Target: {target_desc}")
    print(f"Batch: {args.batch}, Precompute: {args.precompute}")
    if args.duration is not None:
        print(f"Duration: {args.duration}s")
    else:
        print(f"Steps: {args.steps}")
    print()

    _run_single(args, score_fn, target_desc)


if __name__ == '__main__':
    main()
