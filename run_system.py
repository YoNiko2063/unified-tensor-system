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
        """Start L2 GSD improvement cycle."""
        gsd_root = os.path.join(_ROOT, 'external', 'get-shit-done')
        if not os.path.isdir(gsd_root):
            gsd_root = _ROOT
        gsd = GSDBridge(self.tensor, gsd_root, self.dev_agent_root)
        self._components['gsd'] = gsd
        print("[L2] GSD improvement cycle starting...")
        gsd.run_autonomous_cycle(max_phases=3)

    def start_explorer(self):
        """Start configuration explorer targeting NAND."""
        try:
            from tensor.explorer import ConfigurationExplorer, ExplorationTarget
            explorer = ConfigurationExplorer(self.tensor)
            target = ExplorationTarget.logic_gate('NAND')
            result = explorer.explore(target, n_configs=50, batch_size=10)
            self._components['explorer'] = explorer
            print(f"[Explorer] NAND search: best={result.best_score:.4f}, "
                  f"configs={result.configs_evaluated}")
        except Exception as e:
            print(f"[Explorer] Skipped: {e}")

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
