"""LearningTrajectory: the system's memory of its own learning.

Records consonance, eigenvalue gaps, and growth regime over time.
Computes velocity (d/dt), acceleration (d²/dt²), and identifies
compounding vs stagnant subspaces. Meta-loss = -mean(acceleration).

FICUTS Layer 1:
  - Task 1.1: lyapunov_energy() — explicit conserved quantity
  - Task 1.2: meta_loss_stable() — damped acceleration with penalty
  - Task 1.3: Write-ahead journal (WAL) + atomic checkpoint
"""
import json
import threading
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PHI = 1.6180339887


@dataclass
class TrajectoryPoint:
    timestamp: float
    consonance: Dict[str, float]
    eigenvalue_gaps: Dict[str, float]
    growth_regime_count: int
    golden_resonance_matrix: List[List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'consonance': self.consonance,
            'eigenvalue_gaps': self.eigenvalue_gaps,
            'growth_regime_count': self.growth_regime_count,
            'golden_resonance_matrix': self.golden_resonance_matrix,
            'metadata': self.metadata,
        }


class LearningTrajectory:
    def __init__(self, window: int = 500,
                 journal_path: Optional[str] = None):
        self.points: List[TrajectoryPoint] = []
        self.window = window
        self._write_lock = threading.Lock()
        self.checkpoint_interval = 100

        # WAL setup (Task 1.3)
        self._journal = None
        self._journal_path: Optional[Path] = None
        if journal_path is not None:
            p = Path(journal_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._journal_path = p
            self._journal = open(str(p), 'a', buffering=1)  # line-buffered

    def record(self, tensor_context: dict):
        """Append a TrajectoryPoint from a tensor_context dict."""
        point = TrajectoryPoint(
            timestamp=tensor_context.get('timestamp', 0.0),
            consonance=dict(tensor_context.get('consonance', {})),
            eigenvalue_gaps=dict(tensor_context.get('eigenvalue_gaps', {})),
            growth_regime_count=sum(
                n.get('count', 0)
                for n in tensor_context.get('growth_nodes', [])),
            golden_resonance_matrix=[
                row if isinstance(row, list) else [row]
                for row in tensor_context.get('golden_resonance_matrix', [])
            ],
        )
        # Lyapunov energy computed from current points before appending (Task 1.1)
        point.metadata['lyapunov_energy'] = self.lyapunov_energy()

        with self._write_lock:
            self.points.append(point)
            if len(self.points) > self.window:
                self.points = self.points[-self.window:]

            # WAL: append immediately then flush (Task 1.3)
            if self._journal is not None:
                self._journal.write(json.dumps(point.to_dict()) + '\n')
                self._journal.flush()

                if len(self.points) % self.checkpoint_interval == 0:
                    self._atomic_checkpoint()

    # ── Task 1.1: Lyapunov Energy ────────────────────────────────────────────

    def lyapunov_energy(self, level: str = 'code') -> float:
        """E = α·position² + β·velocity² - γ·damping.

        Decreases monotonically toward equilibrium.
        β = φ⁻¹ = 0.618 (golden damping ratio).
        """
        positions = [p.consonance.get(level, 0.0)
                     for p in self.points[-10:]]
        if not positions:
            return 0.0
        vel = self.consonance_velocity(level)

        alpha, beta, gamma = 1.0, 0.618, 0.1
        E_pos = alpha * float(np.mean(np.square(positions)))
        E_vel = beta * vel ** 2
        E_damp = gamma * abs(vel)
        return E_pos + E_vel - E_damp

    # ── Task 1.2: Damped Acceleration Meta-Loss ───────────────────────────────

    def meta_loss_stable(self) -> float:
        """Stable meta-loss: -acceleration + penalty for instability.

        Penalty fires when:
          - velocity ≤ 0 (non-monotonic growth)
          - variance > 0.01 (oscillating)
          - |ΔE/E| > 5% (energy drift)
        """
        accel = self.consonance_acceleration('code')
        vel = self.consonance_velocity('code')
        var = np.var([p.consonance.get('code', 0.0)
                      for p in self.points[-20:]]) if len(self.points) >= 2 else 0.0

        penalty = 0.0
        if vel <= 0:
            penalty += 10.0
        if var > 0.01:
            penalty += var * 100

        if len(self.points) > 10:
            E_prev = self.points[-10].metadata.get('lyapunov_energy', 0)
            E_curr = self.lyapunov_energy()
            if abs((E_curr - E_prev) / max(abs(E_prev), 1e-9)) > 0.05:
                penalty += abs(E_curr - E_prev) * 10

        return -accel + penalty

    # ── Task 1.3: Write-Ahead Journal ────────────────────────────────────────

    def _atomic_checkpoint(self):
        """Write last 500 points to .json via temp-then-rename (atomic)."""
        if self._journal_path is None:
            return
        temp = self._journal_path.with_suffix('.tmp')
        data = [p.to_dict() for p in self.points[-500:]]
        temp.write_text(json.dumps(data))
        temp.replace(self._journal_path.with_suffix('.json'))  # atomic on POSIX

    def recover_from_wal(self, wal_path: str):
        """Rebuild trajectory from WAL after crash. Stops at first corrupt line."""
        recovered = []
        with open(wal_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    recovered.append(TrajectoryPoint(
                        timestamp=d['timestamp'],
                        consonance=d['consonance'],
                        eigenvalue_gaps=d['eigenvalue_gaps'],
                        growth_regime_count=d['growth_regime_count'],
                        golden_resonance_matrix=d['golden_resonance_matrix'],
                        metadata=d.get('metadata', {}),
                    ))
                except (json.JSONDecodeError, KeyError):
                    break  # stop at first corrupt line
        # Trim to window
        self.points = recovered[-self.window:] if len(recovered) > self.window \
            else recovered

    def _consonance_series(self, level: str, n: int = 10) -> List[float]:
        """Last n consonance values for a level."""
        vals = []
        for p in self.points[-(n + 1):]:
            v = p.consonance.get(level)
            if v is not None:
                vals.append(v)
        return vals

    def consonance_velocity(self, level: str) -> float:
        """d(consonance)/dt over last 10 points."""
        series = self._consonance_series(level, 10)
        if len(series) < 2:
            return 0.0
        diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
        return float(np.mean(diffs))

    def consonance_acceleration(self, level: str) -> float:
        """d²(consonance)/dt² — is the system compounding?"""
        series = self._consonance_series(level, 20)
        if len(series) < 3:
            return 0.0
        diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
        d2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        return float(np.mean(d2))

    def compounding_subspaces(self) -> List[str]:
        """Levels where acceleration > 0."""
        levels = set()
        for p in self.points:
            levels.update(p.consonance.keys())
        return [l for l in sorted(levels)
                if self.consonance_acceleration(l) > 0]

    def stagnant_subspaces(self, threshold: int = 10) -> List[str]:
        """Levels where velocity ≈ 0 for > threshold points."""
        levels = set()
        for p in self.points:
            levels.update(p.consonance.keys())
        result = []
        for l in sorted(levels):
            series = self._consonance_series(l, threshold)
            if len(series) >= threshold:
                v = self.consonance_velocity(l)
                if abs(v) < 1e-6:
                    result.append(l)
        return result

    def phi_conjugate_target(self, level: str) -> float:
        """When stagnant: target = equilibrium + PHI * |deviation|."""
        series = self._consonance_series(level, 20)
        if len(series) < 2:
            return 0.0
        eq = float(np.mean(series))
        dev = abs(series[-1] - eq)
        return eq + PHI * dev

    def meta_loss(self) -> float:
        """Negative mean acceleration across all levels. Minimize = maximize learning."""
        levels = set()
        for p in self.points:
            levels.update(p.consonance.keys())
        if not levels:
            return 0.0
        accels = [self.consonance_acceleration(l) for l in levels]
        return float(-np.mean(accels))

    def save(self, path: str):
        """Serialize trajectory to disk."""
        with open(path, 'w') as f:
            json.dump([p.to_dict() for p in self.points], f)

    def load(self, path: str):
        """Load trajectory from disk. Handles old files without metadata."""
        with open(path) as f:
            data = json.load(f)
        self.points = []
        for d in data:
            self.points.append(TrajectoryPoint(
                timestamp=d['timestamp'],
                consonance=d['consonance'],
                eigenvalue_gaps=d['eigenvalue_gaps'],
                growth_regime_count=d['growth_regime_count'],
                golden_resonance_matrix=d['golden_resonance_matrix'],
                metadata=d.get('metadata', {}),
            ))
