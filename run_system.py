#!/usr/bin/env python3
"""Run the full unified tensor system autonomously.

Starts:
  1. RealtimeFeed (L0 market data, background thread)
  2. NeuralBridge continuous loop (L1, background thread)
  3. HardwareProfiler refresh every 5 minutes (L3)
  4. GSD autonomous improvement cycle (L2, foreground)
  5. Explorer (NAND target, background)

Prints tensor snapshot every N seconds.
All logs to tensor/logs/

Stop with Ctrl+C — graceful shutdown, saves state.
"""
import argparse
import os
import sys
import time
import signal
import threading

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))

from tensor.core import UnifiedTensor
from tensor.observer import TensorObserver
from tensor.market_graph import MarketGraph
from tensor.hardware_profiler import HardwareProfiler
from tensor.neural_bridge import NeuralBridge
from tensor.realtime_feed import RealtimeFeed
from tensor.gsd_bridge import GSDBridge
from tensor.context_stream import TensorContextStream


class SystemRunner:
    """Orchestrates all tensor subsystems."""

    def __init__(self, snapshot_interval: int = 60,
                 dev_agent_root: str = 'dev-agent'):
        self.snapshot_interval = snapshot_interval
        self.dev_agent_root = os.path.join(_ROOT, dev_agent_root)
        self._shutdown = threading.Event()
        self._components = {}

        # Build tensor
        self.tensor = UnifiedTensor(n_levels=4, max_nodes='auto')
        self.observer = TensorObserver(self.tensor)

    def _setup_l3_hardware(self):
        """Profile hardware and populate L3."""
        profiler = HardwareProfiler()
        profile = profiler.profile()
        mna = profiler.to_mna(profile)
        self.tensor.update_level(3, mna, t=time.time())
        self._components['profiler'] = profiler
        print(f"[L3] Hardware profiled: {profile.cpu_model or 'unknown'}, "
              f"{profile.cpu_cores}c/{profile.cpu_threads}t, "
              f"{profile.ram_total_gb:.1f}GB RAM")

    def _refresh_l3_loop(self, interval: int = 300):
        """Refresh L3 hardware data periodically."""
        while not self._shutdown.is_set():
            self._shutdown.wait(interval)
            if self._shutdown.is_set():
                break
            profiler = self._components.get('profiler')
            if profiler:
                mna = profiler.to_mna()
                self.tensor.update_level(3, mna, t=time.time())

    def start_feed(self):
        """Start L0 real-time market feed."""
        mg = MarketGraph.mock_live(n_tickers=10)
        feed = RealtimeFeed(
            self.tensor, mg, sources=['mock'], update_interval=2.0)
        feed.start()
        self._components['feed'] = feed
        print(f"[L0] Market feed started: {mg.n_tickers} tickers (mock)")

    def start_neural(self):
        """Start L1 neural continuous loop."""
        bridge = NeuralBridge(self.tensor, n_neurons=16)
        bridge.update_tensor(t=time.time())
        bridge.run_continuous(interval_seconds=5.0)
        self._components['neural'] = bridge
        print("[L1] Neural bridge started: 16 neurons, 5s interval")

    def start_improve(self):
        """Start L2 GSD improvement cycle in a background thread (loops forever)."""
        gsd_root = os.path.join(_ROOT, 'external', 'get-shit-done')
        if not os.path.isdir(gsd_root):
            gsd_root = _ROOT
        gsd = GSDBridge(self.tensor, gsd_root, self.dev_agent_root)
        self._components['gsd'] = gsd

        def _gsd_loop():
            cycle = 0
            while not self._shutdown.is_set():
                cycle += 1
                print(f"\n[GSD] === Cycle {cycle} ===")
                try:
                    gsd.run_autonomous_cycle(max_phases=3)
                except Exception as e:
                    print(f"[GSD] Cycle {cycle} error: {e}")
                # Cool down between cycles
                self._shutdown.wait(60)

        t = threading.Thread(target=_gsd_loop, daemon=True, name='gsd-loop')
        t.start()
        print("[L2] GSD improvement loop started (3 phases per cycle, 60s cooldown)")

    def start_context_stream(self):
        """Start TensorContextStream (Thread 4): publishes JSON to /tmp/tensor_context every 5s."""
        stream = TensorContextStream(self.tensor, interval=5.0)
        stream.start()
        self._components['context_stream'] = stream
        print("[Context] TensorContextStream started: /tmp/tensor_context every 5s")

    def start_explorer(self):
        """Start configuration explorer in background thread.

        Precomputes manifold, loads any previous checkpoint (resuming from
        already-solved points), then explores forever. All results —
        including poor predictions — are checkpointed to disk so future
        generations can reuse them as precomputed ground truth.
        """
        def _explorer_loop():
            try:
                from tensor.explorer import ConfigurationExplorer, ExplorerConfig
                config = ExplorerConfig(
                    target='nand',
                    n_precompute=5000,
                    batch_size=128,
                    checkpoint_interval=500,
                    stats_interval=60.0,
                    log_dir=os.path.join(_ROOT, 'tensor', 'logs'),
                )
                explorer = ConfigurationExplorer(config)

                # Resume from previous checkpoint if it exists
                if explorer.load_checkpoint():
                    prev = explorer.results.count
                    print(f"[Explorer] Resumed {prev:,} previously solved points")
                else:
                    print("[Explorer] No checkpoint found, starting fresh")

                # Precompute manifold grid
                print("[Explorer] Precomputing manifold...")
                explorer.precompute(progress=True)

                self._components['explorer'] = explorer

                # Run forever — all points stored, all checkpointed
                print("[Explorer] Running indefinitely (checkpoints every 500 steps)")
                while not self._shutdown.is_set():
                    explorer.run_step()
                    if explorer._step % config.checkpoint_interval == 0:
                        explorer._checkpoint()
                        print(f"[Explorer] Step {explorer._step}: "
                              f"{explorer.results.count:,} points stored, "
                              f"best={explorer.results.best_score:.4f}")

                # Final checkpoint on shutdown
                explorer._checkpoint()
                print(f"[Explorer] Shutdown: saved {explorer.results.count:,} points")

            except Exception as e:
                print(f"[Explorer] Error: {e}")

        t = threading.Thread(target=_explorer_loop, daemon=True, name='explorer')
        t.start()
        print("[Explorer] Background thread started")

    def print_snapshot(self):
        """Print current tensor state."""
        try:
            md = self.observer.snapshot_markdown()
            print("\n" + "=" * 60)
            print(md)
            print("=" * 60)
            self.observer.log_snapshot()
        except Exception as e:
            print(f"[Snapshot error] {e}")

    def run(self, improve=False, explore=False, feed=False, run_all=False):
        """Main run loop."""
        print("=" * 60)
        print("UNIFIED TENSOR SYSTEM")
        print("=" * 60)

        # Always set up L3
        self._setup_l3_hardware()

        # L3 refresh thread
        l3_thread = threading.Thread(
            target=self._refresh_l3_loop, daemon=True, name='l3-refresh')
        l3_thread.start()

        if feed or run_all:
            self.start_feed()

        if run_all:
            self.start_neural()

        # Start context stream (always on — continuous ambient signaling)
        self.start_context_stream()

        # Initial snapshot
        self.print_snapshot()

        if explore or run_all:
            self.start_explorer()

        if improve or run_all:
            self.start_improve()

        # Snapshot loop
        print(f"\n[System] Snapshot every {self.snapshot_interval}s. Ctrl+C to stop.")
        try:
            while not self._shutdown.is_set():
                self._shutdown.wait(self.snapshot_interval)
                if not self._shutdown.is_set():
                    self.print_snapshot()
        except KeyboardInterrupt:
            pass

        self.shutdown()

    def shutdown(self):
        """Graceful shutdown."""
        print("\n[System] Shutting down...")
        self._shutdown.set()

        feed = self._components.get('feed')
        if feed:
            feed.stop()
            print("[L0] Feed stopped")

        neural = self._components.get('neural')
        if neural:
            neural.stop_continuous()
            print("[L1] Neural bridge stopped")

        ctx_stream = self._components.get('context_stream')
        if ctx_stream:
            ctx_stream.stop()
            print("[Context] TensorContextStream stopped")

        # Final snapshot
        self.observer.log_snapshot()
        print("[System] Final state logged. Goodbye.")


def main():
    parser = argparse.ArgumentParser(
        description='Run the unified tensor system')
    parser.add_argument('--improve', action='store_true',
                        help='Run GSD autonomous improvement cycle')
    parser.add_argument('--explore', action='store_true',
                        help='Run NAND/XOR explorer')
    parser.add_argument('--feed', action='store_true',
                        help='Connect real-time market feeds')
    parser.add_argument('--all', action='store_true',
                        help='Run everything')
    parser.add_argument('--snapshot-interval', type=int, default=60,
                        help='Seconds between snapshots')
    parser.add_argument('--dev-agent-root', type=str, default='dev-agent',
                        help='Path to dev-agent codebase')

    args = parser.parse_args()

    runner = SystemRunner(
        snapshot_interval=args.snapshot_interval,
        dev_agent_root=args.dev_agent_root,
    )

    # Handle signals
    def sig_handler(signum, frame):
        runner.shutdown()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    runner.run(
        improve=args.improve,
        explore=args.explore,
        feed=args.feed,
        run_all=args.all,
    )


if __name__ == '__main__':
    main()
