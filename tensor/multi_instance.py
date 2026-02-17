"""
FICUTS Layer 10: Multi-Instance Coordination

Classes:
  - MultiInstanceCoordinator : spawns/manages child worker processes   (Task 10.1)
  - InstanceWorker           : child process exploring HDV region       (Task 10.1)
  - IsometricTransfer        : geometry-preserving coordinate transfer  (Task 10.2)
  - GlobalManifoldAggregator : combine + dedup discoveries globally     (Task 10.3)
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Callable, Dict, List, Optional

import numpy as np


# ── Task 10.1: Spawning & Management ──────────────────────────────────────────

class MultiInstanceCoordinator:
    """
    Parent instance manages child InstanceWorker processes.
    Each child explores a different HDV region (chart).
    """

    def __init__(self, max_instances: int = 4):
        self.max_instances = max_instances
        self.instances: List[InstanceWorker] = []
        self.result_queue: mp.Queue = mp.Queue()
        self.chart_centers: Dict[int, np.ndarray] = {}

    def should_spawn_child(self, ignorance_map: np.ndarray,
                           learning_priority: np.ndarray) -> bool:
        if len(self.instances) >= self.max_instances:
            return False
        combined = ignorance_map * learning_priority
        max_idx = int(np.argmax(combined))
        for center in self.chart_centers.values():
            if len(center) == len(combined) and np.linalg.norm(combined - center) < 0.5:
                return False
        return float(combined[max_idx]) > 0.3

    def spawn_child(self, chart_center: np.ndarray,
                    exploration_radius: float = 0.5) -> int:
        instance_id = len(self.instances)
        child = InstanceWorker(
            instance_id=instance_id,
            chart_center=chart_center,
            radius=exploration_radius,
            result_queue=self.result_queue,
        )
        child.start()
        self.instances.append(child)
        self.chart_centers[instance_id] = chart_center
        return instance_id

    def collect_results(self, timeout: float = 1.0) -> List[Dict]:
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get(timeout=timeout))
            except Exception:
                break
        return results

    def shutdown_all(self):
        for inst in self.instances:
            inst.terminate()
            inst.join()
        self.instances.clear()
        self.chart_centers.clear()


class InstanceWorker(mp.Process):
    """Child instance: samples a local HDV neighbourhood and reports discoveries."""

    def __init__(self, instance_id: int, chart_center: np.ndarray,
                 radius: float, result_queue: mp.Queue):
        super().__init__()
        self.instance_id = instance_id
        self.chart_center = chart_center
        self.radius = radius
        self.result_queue = result_queue

    def run(self):
        samples = self._sample_neighborhood(100)
        for sample in samples:
            value = self._evaluate_sample(sample)
            if value > 0.7:
                self._report_discovery(sample, value)
        self.result_queue.put({
            'instance_id': self.instance_id,
            'type': 'complete',
            'samples_evaluated': len(samples),
        })

    def _sample_neighborhood(self, n: int) -> List[np.ndarray]:
        return [
            self.chart_center + np.random.randn(len(self.chart_center)) * self.radius
            for _ in range(n)
        ]

    def _evaluate_sample(self, point: np.ndarray) -> float:
        return float(np.random.rand())

    def _report_discovery(self, point: np.ndarray, value: float):
        self.result_queue.put({
            'instance_id': self.instance_id,
            'type': 'discovery',
            'point': point.tolist(),
            'value': value,
            'timestamp': time.time(),
        })


# ── Task 10.2: Isometric Transfer ─────────────────────────────────────────────

class IsometricTransfer:
    """Transfer discoveries between instance coordinate systems (affine translation)."""

    def __init__(self):
        self._transitions: Dict[tuple, Callable] = {}

    def compute_transition(self, center_src: np.ndarray,
                           center_dst: np.ndarray) -> Callable:
        def fn(point: np.ndarray) -> np.ndarray:
            return point + (center_dst - center_src)
        return fn

    def transfer_discovery(self, discovery: Dict,
                           src_center: np.ndarray,
                           dst_center: np.ndarray) -> Dict:
        key = (id(src_center), id(dst_center))
        if key not in self._transitions:
            self._transitions[key] = self.compute_transition(src_center, dst_center)
        point_dst = self._transitions[key](np.array(discovery['point']))
        return {
            'point': point_dst.tolist(),
            'value': discovery['value'],
            'source_instance': discovery['instance_id'],
            'transferred': True,
        }

    def verify_isometry(self, points_src: List[np.ndarray],
                        center_src: np.ndarray, center_dst: np.ndarray,
                        tolerance: float = 0.1) -> bool:
        if len(points_src) < 2:
            return True
        fn = self.compute_transition(center_src, center_dst)
        pts_dst = [fn(p) for p in points_src]
        for i in range(len(points_src)):
            for j in range(i + 1, len(points_src)):
                d_src = np.linalg.norm(points_src[i] - points_src[j])
                d_dst = np.linalg.norm(pts_dst[i] - pts_dst[j])
                if abs(d_src - d_dst) > tolerance:
                    return False
        return True


# ── Task 10.3: Global Aggregation ─────────────────────────────────────────────

class GlobalManifoldAggregator:
    """Combine discoveries from all instances into unified reference frame."""

    def __init__(self):
        self.transfer = IsometricTransfer()
        self.global_discoveries: List[Dict] = []

    def aggregate(self, instance_discoveries: List[Dict],
                  chart_centers: Dict[int, np.ndarray],
                  reference_instance: int = 0) -> List[Dict]:
        ref_center = chart_centers[reference_instance]
        transferred = []

        for disc in instance_discoveries:
            if disc.get('type') != 'discovery':
                continue
            iid = disc['instance_id']
            if iid == reference_instance:
                transferred.append(disc)
            else:
                transferred.append(
                    self.transfer.transfer_discovery(
                        disc, chart_centers[iid], ref_center
                    )
                )

        unique = self._remove_duplicates(transferred, threshold=0.1)
        weighted = sorted(unique, key=lambda d: d['value'], reverse=True)
        self.global_discoveries.extend(weighted)
        return weighted

    def _remove_duplicates(self, discoveries: List[Dict],
                            threshold: float = 0.1) -> List[Dict]:
        unique: List[Dict] = []
        for disc in discoveries:
            pt = np.array(disc['point'])
            is_dup = False
            for i, existing in enumerate(unique):
                if np.linalg.norm(pt - np.array(existing['point'])) < threshold:
                    is_dup = True
                    if disc['value'] > existing['value']:
                        unique[i] = disc
                    break
            if not is_dup:
                unique.append(disc)
        return unique

    def get_top_discoveries(self, n: int = 10) -> List[Dict]:
        return sorted(self.global_discoveries,
                      key=lambda d: d['value'], reverse=True)[:n]
