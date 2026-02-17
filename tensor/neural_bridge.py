"""Bridge between UnifiedTensor L1 and ECEMath neural_ode.

Runs a PhysicsInformedSystem forward pass, builds L1 MNA
(nodes=neurons, edges=weights, C=membrane capacitance, G=synaptic conductance),
updates tensor L1, and supports coarsen/lift chains across levels.
"""
import sys
import os
import numpy as np
from typing import Optional, List, Tuple

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.matrix import MNASystem
from core.neural_ode import PhysicsInformedSystem, numpy_mlp
from tensor.core import UnifiedTensor


class NeuralBridge:
    """Connects PhysicsInformedSystem to tensor L1 (neural layer).

    C·v̇ + G·v + h_NN(v) = u(t)

    C → membrane capacitance (storage)
    G → synaptic conductance (dissipation)
    Nodes = neurons, edges = synaptic weights from G off-diagonals.
    """

    def __init__(self, tensor: UnifiedTensor,
                 n_neurons: int = 16,
                 layer_sizes: Optional[List[int]] = None,
                 seed: int = 42):
        self.tensor = tensor
        self.n_neurons = n_neurons
        self.seed = seed

        # Build physics-informed system
        rng = np.random.default_rng(seed)

        # C: membrane capacitance — diagonal dominant, small coupling
        C = np.eye(n_neurons) * (0.5 + 0.5 * rng.random(n_neurons))
        for i in range(n_neurons - 1):
            c = rng.random() * 0.1
            C[i, i + 1] -= c
            C[i + 1, i] -= c
            C[i, i] += c
            C[i + 1, i + 1] += c

        # G: synaptic conductance — sparse connectivity
        G = np.zeros((n_neurons, n_neurons))
        n_synapses = min(n_neurons * 3, n_neurons * (n_neurons - 1) // 2)
        for _ in range(n_synapses):
            i, j = rng.integers(0, n_neurons, size=2)
            if i != j:
                g = rng.random() * 2.0
                G[i, j] -= g
                G[j, i] -= g
                G[i, i] += g
                G[j, j] += g

        # Diagonal loading for stability
        G += 1e-4 * np.eye(n_neurons)
        C += 1e-4 * np.eye(n_neurons)

        self._C = C
        self._G = G

        # Build h_NN (nonlinear term)
        if layer_sizes is None:
            layer_sizes = [n_neurons, 32, n_neurons]
        self._forward, self._jacobian, self._params = numpy_mlp(
            layer_sizes, activation='tanh', seed=seed)

        def h_model(v):
            return self._forward(v, self._params)

        def h_jac(v):
            return self._jacobian(v, self._params)

        self.system = PhysicsInformedSystem(C, G, h_model, h_jac)
        self._last_state: Optional[np.ndarray] = None
        self._last_mna: Optional[MNASystem] = None

    def to_mna(self) -> MNASystem:
        """Build MNA from current neural network state."""
        n = self.n_neurons
        node_map = {i: i for i in range(n)}
        mna = MNASystem(
            C=self._C.copy(), G=self._G.copy(),
            n_nodes=n, n_branches=0, n_total=n,
            node_map=node_map, branch_map={}, branch_info=[],
        )
        self._last_mna = mna
        return mna

    def forward(self, v0: Optional[np.ndarray] = None,
                dt: float = 0.01, steps: int = 10,
                t0: float = 0.0) -> np.ndarray:
        """Run neural_ode forward pass and return final state.

        Uses simple Euler integration of C·v̇ = -G·v - h(v) + u(t).
        """
        n = self.n_neurons
        if v0 is None:
            v0 = np.zeros(n)

        v = v0.copy()
        C_inv = np.linalg.pinv(self._C)

        for step in range(steps):
            t = t0 + step * dt
            h = self.system.nonlinear(v, t)
            u = self.system.input_vector(t)
            # C·v̇ = -G·v - h(v) + u(t)
            v_dot = C_inv @ (-self._G @ v - h + u)
            v = v + dt * v_dot

        self._last_state = v.copy()
        return v

    def update_tensor(self, t: float = 0.0,
                      v0: Optional[np.ndarray] = None,
                      run_forward: bool = True):
        """Run forward pass and update tensor L1.

        Args:
            t: Timestamp for tensor update
            v0: Initial state (default: zeros or last state)
            run_forward: If True, run forward pass first
        """
        if run_forward:
            if v0 is None and self._last_state is not None:
                v0 = self._last_state
            self.forward(v0=v0, t0=t)

        mna = self.to_mna()
        self.tensor.update_level(1, mna, t=t)

        # Set state vector
        if self._last_state is not None:
            self.tensor.set_state(1, self._last_state)

    def coarsen_chain(self, levels: List[int]) -> list:
        """Coarsen through a chain of levels.

        E.g. coarsen_chain([2, 1, 0]) applies φ: L2→L1, then φ: L1→L0.
        Returns list of CoarseGrainResults.
        """
        results = []
        for i in range(len(levels) - 1):
            fine, coarse = levels[i], levels[i + 1]
            result = self.tensor.coarsen_to(fine, coarse)
            results.append(result)
        return results

    def lift_chain(self, levels: List[int],
                   x_start: np.ndarray) -> np.ndarray:
        """Lift through a chain of levels.

        E.g. lift_chain([0, 1, 2], x_L0) lifts L0→L1→L2.
        Returns final lifted state vector.
        """
        x = x_start.copy()
        for i in range(len(levels) - 1):
            coarse, fine = levels[i], levels[i + 1]
            x = self.tensor.lift_from(coarse, fine, x)
        return x

    @property
    def state(self) -> Optional[np.ndarray]:
        return self._last_state
