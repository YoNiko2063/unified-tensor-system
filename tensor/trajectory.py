"""LearningTrajectory: the system's memory of its own learning.

Records consonance, eigenvalue gaps, and growth regime over time.
Computes velocity (d/dt), acceleration (d²/dt²), and identifies
compounding vs stagnant subspaces. Meta-loss = -mean(acceleration).
"""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PHI = 1.6180339887


@dataclass
class TrajectoryPoint:
    timestamp: float
    consonance: Dict[str, float]
    eigenvalue_gaps: Dict[str, float]
    growth_regime_count: int
    golden_resonance_matrix: List[List[float]]


class LearningTrajectory:
    def __init__(self, window: int = 500):
        self.points: List[TrajectoryPoint] = []
        self.window = window

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
        self.points.append(point)
        if len(self.points) > self.window:
            self.points = self.points[-self.window:]

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
        data = []
        for p in self.points:
            data.append({
                'timestamp': p.timestamp,
                'consonance': p.consonance,
                'eigenvalue_gaps': p.eigenvalue_gaps,
                'growth_regime_count': p.growth_regime_count,
                'golden_resonance_matrix': p.golden_resonance_matrix,
            })
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load trajectory from disk."""
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
            ))
