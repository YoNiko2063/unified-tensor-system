"""TensorContextStream: continuous ambient context for dev-agent and other consumers.

Publishes JSON to /tmp/tensor_context every 5 seconds. Replaces discrete
task dispatch with continuous signaling. Dev-agent reads this stream as
ambient context via --context-stream /tmp/tensor_context flag.

Payload:
  eigenvalue_gaps (all levels), phi_weighted FIM priorities, regime,
  consonance, growth_nodes, stress_nodes, golden_resonance_matrix.
"""
import json
import os
import sys
import time
import threading
import numpy as np
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ECEMATH_SRC = os.path.join(_ROOT, 'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

from tensor.core import UnifiedTensor, LEVEL_NAMES
from tensor.math_connections import fisher_guided_planning, detect_regime
from tensor.trajectory import LearningTrajectory
from tensor.agent_network import PredictiveLayer
from tensor.domain_fibers import FiberBundle


class TensorContextStream:
    """Publishes tensor state as JSON for continuous ambient signaling."""

    def __init__(self, tensor: UnifiedTensor,
                 output_path: str = '/tmp/tensor_context',
                 interval: float = 5.0,
                 trajectory: Optional['LearningTrajectory'] = None,
                 predictive: Optional['PredictiveLayer'] = None,
                 fiber_bundle: Optional['FiberBundle'] = None):
        self.tensor = tensor
        self.output_path = output_path
        self.interval = interval
        self.trajectory = trajectory
        self.predictive = predictive
        self.fiber_bundle = fiber_bundle
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def _build_payload(self) -> dict:
        """Build the JSON payload from current tensor state."""
        payload = {
            'timestamp': time.time(),
            'eigenvalue_gaps': {},
            'consonance': {},
            'growth_nodes': [],
            'stress_nodes': [],
            'golden_resonance_matrix': [],
        }

        # Eigenvalue gaps and consonance for all populated levels
        for l in range(self.tensor.n_levels):
            mna = self.tensor._mna.get(l)
            if mna is None:
                continue
            name = LEVEL_NAMES.get(l, f'L{l}')
            payload['eigenvalue_gaps'][name] = round(
                self.tensor.eigenvalue_gap(l), 4)

            sig = self.tensor.harmonic_signature(l)
            payload['consonance'][name] = round(sig.consonance_score, 4)

            # Growth regime nodes
            growth_count = getattr(sig, 'growth_regime_count', 0)
            if growth_count > 0:
                payload['growth_nodes'].append({
                    'level': name,
                    'count': growth_count,
                })

        # FIM priorities with phi_weights (L2)
        try:
            guidance = fisher_guided_planning(self.tensor, level=2, top_k=5)
            payload['fim_priorities'] = {
                'indices': guidance.priority_indices,
                'phi_weights': guidance.phi_weights.tolist()
                    if guidance.phi_weights is not None else [],
                'condition_number': round(guidance.condition_number, 2),
            }
        except Exception:
            payload['fim_priorities'] = {
                'indices': [], 'phi_weights': [], 'condition_number': 1.0}

        # Regime detection
        try:
            regime = detect_regime(self.tensor, level=2)
            payload['regime'] = {
                'current': regime.current_regime,
                'transition_probability': round(regime.transition_probability, 4),
                'should_pause': regime.should_pause,
            }
        except Exception:
            payload['regime'] = {
                'current': 0, 'transition_probability': 0.0,
                'should_pause': False}

        # Stress nodes: levels with high phase transition risk
        for l in range(self.tensor.n_levels):
            mna = self.tensor._mna.get(l)
            if mna is None:
                continue
            risk = self.tensor.phase_transition_risk(l)
            if risk > 0.6:
                payload['stress_nodes'].append({
                    'level': LEVEL_NAMES.get(l, f'L{l}'),
                    'risk': round(risk, 4),
                })

        # Golden resonance matrix
        try:
            grm = self.tensor.golden_resonance_matrix()
            if grm.size > 0:
                payload['golden_resonance_matrix'] = grm.tolist()
        except Exception:
            pass

        # Trajectory fields (L1 extension)
        if self.trajectory is not None:
            payload['consonance_velocity'] = {}
            payload['consonance_acceleration'] = {}
            for name in payload['consonance']:
                payload['consonance_velocity'][name] = round(
                    self.trajectory.consonance_velocity(name), 6)
                payload['consonance_acceleration'][name] = round(
                    self.trajectory.consonance_acceleration(name), 6)
            payload['compounding_subspaces'] = self.trajectory.compounding_subspaces()

        # Predictive layer fields (L3 extension)
        if self.predictive is not None:
            payload['ignorance_map'] = {
                k: round(v, 4)
                for k, v in self.predictive.ignorance_map().items()}
            payload['learning_priority'] = self.predictive.learning_priority()

        # Fiber bundle fields (L4 extension)
        if self.fiber_bundle is not None:
            payload['fiber_resonance_matrix'] = self.fiber_bundle.fiber_resonance_matrix()
            payload['universal_patterns_found'] = len(
                self.fiber_bundle.universal_patterns())

        return payload

    def _publish_loop(self):
        """Background loop: publish payload every interval seconds."""
        while not self._shutdown.is_set():
            try:
                payload = self._build_payload()
                # Atomic write: write to temp then rename
                tmp_path = self.output_path + '.tmp'
                with open(tmp_path, 'w') as f:
                    json.dump(payload, f, indent=2, default=str)
                os.replace(tmp_path, self.output_path)
            except Exception as e:
                # Don't crash the thread on transient errors
                pass
            self._shutdown.wait(self.interval)

    def start(self):
        """Start the context stream background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._publish_loop, daemon=True,
            name='tensor-context-stream')
        self._thread.start()

    def stop(self):
        """Stop the context stream."""
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def publish_once(self) -> dict:
        """Publish a single snapshot (for testing/manual use)."""
        payload = self._build_payload()
        tmp_path = self.output_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, self.output_path)
        return payload
