"""Memory-first configuration space explorer.

Precomputes a grid of MNA matrices, their eigendecompositions, and
harmonic signatures in RAM. Exploration navigates this cached manifold
via batch-scored neighbors — no redundant computation.

Designed for 48GB RAM machines: holds 10k+ configurations, 500k+ results,
and a sparse similarity matrix all in memory.

Level-aware: each ExplorationTarget knows which tensor level it operates
on (L0=market, L1=neural, L2=code, L3=hw, L-1=cross-level) and the
physical semantics of eigenvalues and ratios at that level.
"""
import os
import sys
import time
import json
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from multiprocessing import Pool, cpu_count

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.matrix import MNASystem
from core.sparse_solver import (
    compute_harmonic_signature, HarmonicSignature,
    consonance_score_from_ratios, harmonic_distance, nearest_consonant,
)

# Control BLAS threading — set before numpy operations in workers
_N_CORES = cpu_count() or 4
_BLAS_THREADS = str(max(1, _N_CORES - 2))


# ═══════════════════════════════════════════════════════════
# ZERO EIGENVALUE HELPER
# ═══════════════════════════════════════════════════════════

def _skip_zero_eigenvalues(sorted_eig: np.ndarray,
                           threshold: float = 1e-10) -> np.ndarray:
    """Remove near-zero eigenvalues (ground node artifacts).

    Args:
        sorted_eig: Eigenvalues sorted descending by magnitude.
        threshold: Values below this are considered zero.

    Returns:
        Sub-array with near-zero values removed. If all are
        near-zero, returns a single-element array [threshold].
    """
    mask = np.abs(sorted_eig) > threshold
    if not np.any(mask):
        return np.array([threshold])
    return sorted_eig[mask]


# ═══════════════════════════════════════════════════════════
# CONFIGURATION SPEC
# ═══════════════════════════════════════════════════════════

@dataclass
class ExplorerConfig:
    """Explorer parameters."""
    n_precompute: int = 10000
    batch_size: int = 256
    max_results: int = 500_000
    ram_limit_gb: float = 40.0
    checkpoint_interval: int = 1000
    stats_interval: float = 30.0
    target: str = 'bandpass'
    log_dir: str = 'tensor/logs'
    resume: bool = False
    score_fn: Optional[Callable] = None  # Override target's default scorer
    duration: Optional[float] = None     # Run for N seconds then stop
    scale_ram: bool = False              # Use RAM scaling in precompute


# ═══════════════════════════════════════════════════════════
# SCORING TARGETS (legacy module-level for backward compat)
# ═══════════════════════════════════════════════════════════

def _score_bandpass(eigvals: np.ndarray, cons_score: float,
                    dom_interval: str) -> float:
    """Score a configuration for bandpass filter behavior.

    Physical basis: MNA eigenvalues are omega^2. For a 2nd-order
    bandpass with Q~10, target ratios are [1.0, 1.1, 1.2].
    """
    if len(eigvals) < 2:
        return 0.0
    sorted_eig = _skip_zero_eigenvalues(np.sort(np.abs(eigvals))[::-1])
    if len(sorted_eig) < 2:
        return 0.0

    # Target ratios for Q=10 bandpass (descending, normalized to λ₀)
    # λ_k/λ₀ = 1/(1+k/Q) for 2nd-order bandpass
    target_ratios = np.array([1.0, 10.0/11.0, 10.0/12.0])
    actual_ratios = sorted_eig[:3] / sorted_eig[0]
    n_compare = min(len(actual_ratios), len(target_ratios))
    ratio_error = np.sum((actual_ratios[:n_compare] - target_ratios[:n_compare])**2)
    ratio_score = np.exp(-ratio_error * 5.0)

    gap = (sorted_eig[0] - sorted_eig[1]) / sorted_eig[0]
    spread = 1.0 / (1.0 + np.log1p(sorted_eig[0] / (sorted_eig[-1] + 1e-30)) / 10.0)

    return float(0.40 * ratio_score + 0.25 * cons_score + 0.20 * gap + 0.15 * spread)


def _score_snn(eigvals: np.ndarray, cons_score: float,
               dom_interval: str) -> float:
    """Score for spiking neural network stability."""
    if len(eigvals) < 2:
        return 0.0
    sorted_eig = _skip_zero_eigenvalues(np.sort(np.abs(eigvals))[::-1])
    if len(sorted_eig) < 2:
        return 0.0

    # Moderate gap (not too wide, not too narrow)
    gap = (sorted_eig[0] - sorted_eig[1]) / sorted_eig[0]
    gap_score = 1.0 - abs(gap - 0.3)

    # Eigenvalue clustering: count near-consonant pairs
    ratios = sorted_eig / sorted_eig[0]
    n_consonant = 0
    for i in range(1, len(ratios)):
        if abs(ratios[i]) > 1e-30:
            r = abs(ratios[i - 1] / ratios[i])
            hd = harmonic_distance(r)
            if hd < 0.3:
                n_consonant += 1
    cluster_score = n_consonant / max(len(ratios) - 1, 1)

    return float(0.4 * cons_score + 0.3 * gap_score + 0.3 * cluster_score)


def _score_custom(eigvals: np.ndarray, cons_score: float,
                  dom_interval: str) -> float:
    """Generic score: maximize consonance and eigenvalue gap."""
    if len(eigvals) < 2:
        return 0.0
    sorted_eig = _skip_zero_eigenvalues(np.sort(np.abs(eigvals))[::-1])
    if len(sorted_eig) < 2:
        return 0.0
    gap = (sorted_eig[0] - sorted_eig[1]) / sorted_eig[0]
    return float(0.5 * cons_score + 0.5 * gap)


_SCORE_FUNCS = {
    'bandpass': _score_bandpass,
    'snn': _score_snn,
    'custom': _score_custom,
}


# ═══════════════════════════════════════════════════════════
# EXPLORATION TARGET (level-aware parameterized scoring)
# ═══════════════════════════════════════════════════════════

class ExplorationTarget:
    """Level-aware parameterized scoring target for the explorer.

    Each target knows which tensor level it operates on and the
    physical semantics of eigenvalues/ratios at that level.

    Instances are callable — use directly as score_fn:
        target = ExplorationTarget.bandpass_filter(freq=1000, Q=10)
        score = target(eigvals, cons_score, dom_interval)
    """

    def __init__(self, name: str, target_ratios, *,
                 tensor_level: int, level_name: str,
                 eigenvalue_semantics: str, ratio_semantics: str,
                 physical_unit: str,
                 n_nodes: int = 0,
                 stability_preference: float = 0.5,
                 physical_constraints: Optional[Dict] = None,
                 ratio_weight: float = 0.40,
                 consonance_weight: float = 0.25,
                 gap_weight: float = 0.20,
                 spread_weight: float = 0.15):
        self.name = name
        # Normalize target ratios: descending, starting from 1.0
        raw = np.asarray(target_ratios, dtype=np.float64)
        raw_sorted = np.sort(raw)[::-1]  # descending
        self.target_ratios = raw_sorted / raw_sorted[0] if raw_sorted[0] > 0 else raw_sorted
        self.tensor_level = tensor_level
        self.level_name = level_name
        self.eigenvalue_semantics = eigenvalue_semantics
        self.ratio_semantics = ratio_semantics
        self.physical_unit = physical_unit
        self.n_nodes = n_nodes
        self.stability_preference = stability_preference
        self.physical_constraints = physical_constraints or {}
        self._ratio_weight = ratio_weight
        self._consonance_weight = consonance_weight
        self._gap_weight = gap_weight
        self._spread_weight = spread_weight

    def __call__(self, eigvals: np.ndarray, cons_score: float,
                 dom_interval: str) -> float:
        """Score eigenvalues against this target's ratio signature."""
        if len(eigvals) < 2:
            return 0.0
        sorted_eig = _skip_zero_eigenvalues(np.sort(np.abs(eigvals))[::-1])
        if len(sorted_eig) < 2:
            return 0.0

        # Ratio match against target
        n_target = len(self.target_ratios)
        actual_ratios = sorted_eig[:n_target] / sorted_eig[0]
        n_compare = min(len(actual_ratios), n_target)
        ratio_error = np.sum(
            (actual_ratios[:n_compare] - self.target_ratios[:n_compare])**2)
        ratio_score = np.exp(-ratio_error * 5.0)

        # Consonance
        cons_bonus = cons_score

        # Gap
        gap = (sorted_eig[0] - sorted_eig[1]) / sorted_eig[0]

        # Spread
        spread = 1.0 / (1.0 + np.log1p(
            sorted_eig[0] / (sorted_eig[-1] + 1e-30)) / 10.0)

        return float(self._ratio_weight * ratio_score +
                     self._consonance_weight * cons_bonus +
                     self._gap_weight * gap +
                     self._spread_weight * spread)

    def __repr__(self):
        return (f"ExplorationTarget({self.name!r}, L{self.tensor_level}"
                f"({self.level_name}), unit={self.physical_unit!r})")

    @classmethod
    def bandpass_filter(cls, freq: float = 1000.0,
                        Q: float = 10.0) -> 'ExplorationTarget':
        """Score target for bandpass filter with center frequency and Q.

        Physical basis: MNA eigenvalues = omega^2 (squared frequencies).
        For 2nd-order bandpass: lambda_k/lambda_1 = 1 + k/Q.
        Q=10 -> target_ratios = [1.0, 1.1, 1.2] (minor_second cluster).
        """
        # Physical: λ_k/λ₀ = 1/(1+k/Q) — descending ratios
        n_ratios = 3
        target_ratios = [1.0 / (1.0 + k / Q) for k in range(n_ratios)]
        return cls(
            name=f'bandpass_f{freq:.0f}_Q{Q:.0f}',
            target_ratios=target_ratios,
            tensor_level=0,
            level_name='market',
            eigenvalue_semantics='squared_frequency',
            ratio_semantics='frequency_spacing',
            physical_unit='Hz\u00b2',
            stability_preference=0.25,
            physical_constraints={'freq': freq, 'Q': Q},
        )

    @classmethod
    def snn_configuration(cls, neurons: int = 16,
                          sparsity: float = 0.8) -> 'ExplorationTarget':
        """Score target for spiking neural network configuration.

        Eigenvalues = synaptic time constants. Ratios = firing rate
        relationships. High sparsity -> fast eigenvalue decay.
        """
        expected_rank = max(2, int(neurons * (1.0 - sparsity)))
        target_ratios = [1.0 / (1.0 + 0.1 * k) for k in range(expected_rank)]
        return cls(
            name=f'snn_n{neurons}_s{sparsity:.1f}',
            target_ratios=target_ratios,
            tensor_level=1,
            level_name='neural',
            eigenvalue_semantics='synaptic_time_constant',
            ratio_semantics='firing_rate_ratio',
            physical_unit='ms\u207b\u00b9',
            n_nodes=neurons,
            stability_preference=0.40,
            consonance_weight=0.40,
            ratio_weight=0.20,
            gap_weight=0.20,
            spread_weight=0.20,
            physical_constraints={
                'neurons': neurons, 'sparsity': sparsity,
                'target_gap': 0.3 / np.sqrt(max(neurons, 1)),
            },
        )

    @classmethod
    def logic_gate(cls, gate_type: str) -> 'ExplorationTarget':
        """Eigenvalue signature targets for boolean logic gates.

        Logic gates live at L1 (neural) because a transistor gate IS
        a biological neuron analog -- same threshold-firing mechanism,
        different substrate.

        Gate signatures:
            NOT:  [1.0, 2.0]                 octave (perfect inversion)
            AND:  [1.0, 0.1, 0.09]           large gap (threshold)
            OR:   [1.0, 0.95, 0.1]           inverted AND
            NAND: [1.0, 1.414, 2.0, 2.828]   tritone stack (universal)
            XOR:  [1.0, 1.5, 1.5, 2.0]       broken symmetry (degenerate)
        """
        GATE_SIGNATURES = {
            'NOT':  {'n_nodes': 2, 'ratios': [1.0, 2.0],
                     'stability': 1.0},
            'AND':  {'n_nodes': 3, 'ratios': [1.0, 0.1, 0.09],
                     'stability': 0.9},
            'OR':   {'n_nodes': 3, 'ratios': [1.0, 0.95, 0.1],
                     'stability': 0.9},
            'NAND': {'n_nodes': 4, 'ratios': [1.0, 1.414, 2.0, 2.828],
                     'stability': 0.5},
            'XOR':  {'n_nodes': 4, 'ratios': [1.0, 1.5, 1.5, 2.0],
                     'stability': 0.3},
        }
        gate_type = gate_type.upper()
        if gate_type not in GATE_SIGNATURES:
            raise ValueError(
                f"Unknown gate: {gate_type}. "
                f"Valid: {list(GATE_SIGNATURES.keys())}")
        sig = GATE_SIGNATURES[gate_type]
        return cls(
            name=f'logic_{gate_type}',
            target_ratios=sig['ratios'],
            tensor_level=1,
            level_name='neural',
            eigenvalue_semantics='switching_threshold',
            ratio_semantics='logic_voltage_ratio',
            physical_unit='V\u00b2',
            n_nodes=sig['n_nodes'],
            stability_preference=sig['stability'],
            ratio_weight=0.50,
            consonance_weight=0.20,
            gap_weight=0.15,
            spread_weight=0.15,
            physical_constraints={
                'gate_type': gate_type,
                'min_resistance': 1e2,
                'max_resistance': 1e6,
                'allow_negative_resistance': False,
            },
        )

    @classmethod
    def code_structure(cls, target_complexity: float = 5.0,
                       target_coupling: float = 0.3) -> 'ExplorationTarget':
        """Score target for code architecture search.

        Eigenvalues = module coupling strengths.
        Ratios = dependency depth relationships.

        High complexity -> more eigenvalue spread (more modes).
        High coupling -> eigenvalues cluster (tighter ratios).

        Use case: dev-agent searches for better code structures
        using the same explorer.
        """
        n_modes = max(2, int(math.ceil(target_complexity)))
        target_ratios = [1.0 / (1.0 + k * target_coupling)
                         for k in range(n_modes)]
        return cls(
            name=f'code_cx{target_complexity:.1f}_cp{target_coupling:.1f}',
            target_ratios=target_ratios,
            tensor_level=2,
            level_name='code',
            eigenvalue_semantics='module_coupling_strength',
            ratio_semantics='dependency_depth_ratio',
            physical_unit='complexity',
            stability_preference=0.5,
            ratio_weight=0.35,
            consonance_weight=0.30,
            gap_weight=0.20,
            spread_weight=0.15,
            physical_constraints={
                'target_complexity': target_complexity,
                'target_coupling': target_coupling,
            },
        )

    @classmethod
    def cross_level_resonance(cls, level_a: int = 0, level_b: int = 2,
                              target_resonance: float = 0.8) -> 'ExplorationTarget':
        """Score target for cross-level harmonic alignment.

        Finds configurations where two tensor levels are maximally
        harmonically aligned. tensor_level = -1 (special cross-level).

        target_resonance (0-1): how close to perfect consonance
        between the two levels' eigenvalue spectra.

        Use case: find configs where market and code are maximally
        harmonically aligned — the highest-dimensional target.
        """
        # Target ratios for cross-level: alternating ratios from both levels
        # At perfect resonance, both levels share the same dominant ratios
        # We encode the target as a geometric series converging at resonance
        n_ratios = 6
        target_ratios = [target_resonance ** (k * 0.5) for k in range(n_ratios)]
        level_names = {0: 'market', 1: 'neural', 2: 'code', 3: 'hw'}
        name_a = level_names.get(level_a, f'L{level_a}')
        name_b = level_names.get(level_b, f'L{level_b}')
        return cls(
            name=f'resonance_L{level_a}_L{level_b}',
            target_ratios=target_ratios,
            tensor_level=-1,
            level_name=f'resonance_{name_a}_{name_b}',
            eigenvalue_semantics='cross_level_resonance',
            ratio_semantics='harmonic_alignment',
            physical_unit='dimensionless',
            stability_preference=0.5,
            ratio_weight=0.35,
            consonance_weight=0.35,
            gap_weight=0.15,
            spread_weight=0.15,
            physical_constraints={
                'level_a': level_a,
                'level_b': level_b,
                'target_resonance': target_resonance,
            },
        )


# ═══════════════════════════════════════════════════════════
# MATRIX GENERATION
# ═══════════════════════════════════════════════════════════

def _generate_chain(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Chain topology: each node connected to neighbors."""
    G = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n - 1):
        g = 0.1 + rng.random() * 9.9  # log-ish spread
        c = 0.01 + rng.random() * 0.99
        G[i, i] += g; G[i+1, i+1] += g
        G[i, i+1] -= g; G[i+1, i] -= g
        C[i, i] += c; C[i+1, i+1] += c
        C[i, i+1] -= c; C[i+1, i] -= c
    G += 1e-6 * np.eye(n)
    C += 1e-6 * np.eye(n)
    return G, C


def _generate_star(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Star topology: node 0 connected to all others."""
    G = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(1, n):
        g = 0.1 + rng.random() * 9.9
        c = 0.01 + rng.random() * 0.99
        G[0, 0] += g; G[i, i] += g
        G[0, i] -= g; G[i, 0] -= g
        C[0, 0] += c; C[i, i] += c
        C[0, i] -= c; C[i, 0] -= c
    G += 1e-6 * np.eye(n)
    C += 1e-6 * np.eye(n)
    return G, C


def _generate_mesh(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Mesh: grid with 4-connectivity, exactly n nodes."""
    G = np.zeros((n, n))
    C = np.zeros((n, n))
    side = max(2, int(np.sqrt(n)))
    for r in range(n):
        for delta in [1, side]:
            j = r + delta
            if j < n:
                g = 0.1 + rng.random() * 9.9
                cv = 0.01 + rng.random() * 0.99
                G[r, r] += g; G[j, j] += g
                G[r, j] -= g; G[j, r] -= g
                C[r, r] += cv; C[j, j] += cv
                C[r, j] -= cv; C[j, r] -= cv
    G += 1e-6 * np.eye(n)
    C += 1e-6 * np.eye(n)
    return G, C


def _generate_random_sparse(n: int, rng: np.random.Generator,
                             density: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Random sparse symmetric topology."""
    G = np.zeros((n, n))
    C = np.zeros((n, n))
    n_edges = max(n - 1, int(n * (n - 1) / 2 * density))
    edges_added = set()
    # Ensure connected: chain backbone
    for i in range(n - 1):
        g = 0.1 + rng.random() * 9.9
        c = 0.01 + rng.random() * 0.99
        G[i, i] += g; G[i+1, i+1] += g
        G[i, i+1] -= g; G[i+1, i] -= g
        C[i, i] += c; C[i+1, i+1] += c
        C[i, i+1] -= c; C[i+1, i] -= c
        edges_added.add((i, i+1))
    # Random additional edges
    for _ in range(n_edges - (n - 1)):
        i, j = rng.integers(0, n, size=2)
        if i == j or (min(i, j), max(i, j)) in edges_added:
            continue
        edges_added.add((min(i, j), max(i, j)))
        g = 0.1 + rng.random() * 9.9
        c = 0.01 + rng.random() * 0.99
        G[i, i] += g; G[j, j] += g
        G[i, j] -= g; G[j, i] -= g
        C[i, i] += c; C[j, j] += c
        C[i, j] -= c; C[j, i] -= c
    G += 1e-6 * np.eye(n)
    C += 1e-6 * np.eye(n)
    return G, C


_TOPOLOGIES = {
    'chain': _generate_chain,
    'star': _generate_star,
    'mesh': _generate_mesh,
    'random': _generate_random_sparse,
}


# ═══════════════════════════════════════════════════════════
# WORKER FUNCTION (for multiprocessing)
# ═══════════════════════════════════════════════════════════

def _precompute_one(args: Tuple) -> dict:
    """Precompute eigendecomposition + harmonic signature for one config.

    Called by Pool.map — must be top-level function.
    """
    idx, n_nodes, topo_name, seed = args
    rng = np.random.default_rng(seed)
    gen_fn = _TOPOLOGIES[topo_name]
    G, C = gen_fn(n_nodes, rng)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    sort_idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Harmonic signature — skip zero eigenvalues for correct ratios
    clean_eig = _skip_zero_eigenvalues(np.sort(np.abs(eigvals))[::-1])
    k = min(len(clean_eig), 10)
    ratios = clean_eig[:k] / clean_eig[0]
    cons_score = float(consonance_score_from_ratios(ratios))

    # Dominant interval
    if len(ratios) > 1 and abs(ratios[1]) > 1e-30:
        r = abs(ratios[0] / ratios[1])
        dom_name, _ = nearest_consonant(r)
    else:
        dom_name = 'unison'

    return {
        'idx': idx,
        'n_nodes': n_nodes,
        'topology': topo_name,
        'seed': seed,
        'eigvals': eigvals,
        'cons_score': cons_score,
        'dom_interval': dom_name,
        'G_flat': G.ravel(),
        'C_flat': C.ravel(),
    }


# ═══════════════════════════════════════════════════════════
# PRECOMPUTED MANIFOLD
# ═══════════════════════════════════════════════════════════

class PrecomputedManifold:
    """In-RAM cache of precomputed MNA configurations.

    Stores all matrices, eigendecompositions, and signatures
    as contiguous numpy arrays for batch operations.
    """

    def __init__(self, max_n: int = 32):
        self.max_n = max_n
        self.n_configs = 0
        # Padded storage (all configs padded to max_n x max_n)
        self._G: Optional[np.ndarray] = None      # (n_configs, max_n, max_n)
        self._C: Optional[np.ndarray] = None      # (n_configs, max_n, max_n)
        self._eigvals: Optional[np.ndarray] = None # (n_configs, max_n)
        self._cons_scores: Optional[np.ndarray] = None  # (n_configs,)
        self._dom_intervals: Optional[List[str]] = None
        self._n_nodes: Optional[np.ndarray] = None  # (n_configs,) actual node counts
        self._topologies: Optional[List[str]] = None
        self._seeds: Optional[np.ndarray] = None

    def build(self, n_precompute: int, n_workers: int = None,
              progress_cb: Callable = None):
        """Precompute configurations in parallel.

        Args:
            n_precompute: Number of configurations to generate.
            n_workers: Pool size (default: cpu_count - 2).
            progress_cb: Called with (n_done, n_total) for progress.
        """
        if n_workers is None:
            n_workers = max(1, _N_CORES - 2)

        # Build task list: structured grid
        node_sizes = [4, 8, 16, 32]
        topo_names = list(_TOPOLOGIES.keys())
        tasks = []
        base_seed = 42
        per_size = n_precompute // len(node_sizes)
        for si, n_nodes in enumerate(node_sizes):
            # Adjust: fewer 32-node configs for RAM efficiency
            count = per_size // 2 if n_nodes == 32 else per_size
            for ti in range(count):
                topo = topo_names[ti % len(topo_names)]
                seed = base_seed + si * 100000 + ti
                tasks.append((len(tasks), n_nodes, topo, seed))

        # Pad to exact count
        while len(tasks) < n_precompute:
            si = len(tasks) % len(node_sizes)
            n_nodes = node_sizes[si]
            topo = topo_names[len(tasks) % len(topo_names)]
            seed = base_seed + 900000 + len(tasks)
            tasks.append((len(tasks), n_nodes, topo, seed))
        tasks = tasks[:n_precompute]

        # Parallel precomputation
        results = []
        if n_workers > 1:
            chunk = max(1, len(tasks) // (n_workers * 4))
            with Pool(n_workers) as pool:
                for i, result in enumerate(pool.imap_unordered(_precompute_one, tasks, chunksize=chunk)):
                    results.append(result)
                    if progress_cb and (i + 1) % 500 == 0:
                        progress_cb(i + 1, len(tasks))
        else:
            for i, task in enumerate(tasks):
                results.append(_precompute_one(task))
                if progress_cb and (i + 1) % 500 == 0:
                    progress_cb(i + 1, len(tasks))

        # Sort by original index
        results.sort(key=lambda r: r['idx'])
        self.n_configs = len(results)

        # Allocate contiguous arrays
        max_n = self.max_n
        self._G = np.zeros((self.n_configs, max_n, max_n), dtype=np.float64)
        self._C = np.zeros((self.n_configs, max_n, max_n), dtype=np.float64)
        self._eigvals = np.zeros((self.n_configs, max_n), dtype=np.float64)
        self._cons_scores = np.zeros(self.n_configs, dtype=np.float64)
        self._dom_intervals = []
        self._n_nodes = np.zeros(self.n_configs, dtype=np.int32)
        self._topologies = []
        self._seeds = np.zeros(self.n_configs, dtype=np.int64)

        for r in results:
            i = r['idx']
            n = r['n_nodes']
            self._n_nodes[i] = n
            self._G[i, :n, :n] = r['G_flat'].reshape(n, n)
            self._C[i, :n, :n] = r['C_flat'].reshape(n, n)
            ne = len(r['eigvals'])
            self._eigvals[i, :ne] = r['eigvals']
            self._cons_scores[i] = r['cons_score']
            self._dom_intervals.append(r['dom_interval'])
            self._topologies.append(r['topology'])
            self._seeds[i] = r['seed']

    def scale_to_ram(self, target_gb: float, n_workers: int = None,
                     progress_cb: Callable = None):
        """Compute how many configs fit in target_gb and build that many.

        Memory per config: 2 * max_n^2 * 8 + max_n * 8 + 20 bytes
        """
        bytes_per_config = (2 * self.max_n**2 * 8 +
                            self.max_n * 8 + 20)
        target_bytes = target_gb * 1024**3
        n_configs = int(target_bytes / bytes_per_config)
        n_configs = min(max(1000, n_configs), 2_000_000)

        print(f"  RAM target: {target_gb:.1f}GB -> {n_configs:,} configs "
              f"({bytes_per_config} bytes/config)")

        self.build(n_configs, n_workers=n_workers, progress_cb=progress_cb)

    def ram_usage_mb(self) -> float:
        """Estimated RAM usage in MB."""
        if self._G is None:
            return 0.0
        total = (self._G.nbytes + self._C.nbytes + self._eigvals.nbytes +
                 self._cons_scores.nbytes + self._n_nodes.nbytes +
                 self._seeds.nbytes)
        return total / (1024 * 1024)

    def get_G(self, idx: int) -> np.ndarray:
        n = self._n_nodes[idx]
        return self._G[idx, :n, :n]

    def get_eigvals(self, idx: int) -> np.ndarray:
        n = self._n_nodes[idx]
        return self._eigvals[idx, :n]


# ═══════════════════════════════════════════════════════════
# RESULTS BUFFER
# ═══════════════════════════════════════════════════════════

class ResultsBuffer:
    """Fixed-size RAM buffer for exploration results.

    Stores eigenvalues + score + config params as structured numpy array.
    Only writes to disk on checkpoint.
    """

    def __init__(self, max_results: int = 500_000, max_n: int = 32):
        self.max_results = max_results
        self.max_n = max_n
        # result_dim = eigenvalues(max_n) + score(1) + params(max_n) + config_idx(1) + step(1)
        self.result_dim = max_n + 1 + max_n + 1 + 1
        self._buffer = np.zeros((max_results, self.result_dim), dtype=np.float64)
        self._count = 0
        self._best_score = -1.0
        self._best_idx = -1

    def add(self, eigvals: np.ndarray, score: float,
            params: np.ndarray, config_idx: int, step: int):
        """Add a single result to the buffer."""
        if self._count >= self.max_results:
            # Evict oldest half
            half = self.max_results // 2
            self._buffer[:half] = self._buffer[half:]
            self._count = half

        row = self._buffer[self._count]
        ne = min(len(eigvals), self.max_n)
        row[:ne] = eigvals[:ne]
        row[self.max_n] = score
        np_len = min(len(params), self.max_n)
        row[self.max_n + 1:self.max_n + 1 + np_len] = params[:np_len]
        row[self.max_n + 1 + self.max_n] = config_idx
        row[self.max_n + 1 + self.max_n + 1 - 1] = step

        if score > self._best_score:
            self._best_score = score
            self._best_idx = self._count

        self._count += 1

    def add_batch(self, eigvals_batch: np.ndarray, scores: np.ndarray,
                  config_indices: np.ndarray, step: int):
        """Add a batch of results efficiently."""
        n_batch = len(scores)
        if self._count + n_batch > self.max_results:
            # Evict oldest half
            half = self.max_results // 2
            self._buffer[:half] = self._buffer[half:]
            self._count = half

        start = self._count
        end = start + n_batch
        ne = min(eigvals_batch.shape[1], self.max_n)
        self._buffer[start:end, :ne] = eigvals_batch[:, :ne]
        self._buffer[start:end, self.max_n] = scores
        self._buffer[start:end, self.max_n + 1 + self.max_n] = config_indices

        best_in_batch = np.argmax(scores)
        if scores[best_in_batch] > self._best_score:
            self._best_score = float(scores[best_in_batch])
            self._best_idx = start + int(best_in_batch)

        self._count = end

    @property
    def count(self) -> int:
        return self._count

    @property
    def best_score(self) -> float:
        return self._best_score

    def scores(self) -> np.ndarray:
        """All scores in buffer."""
        return self._buffer[:self._count, self.max_n].copy()

    def best_configurations(self, top_k: int = 10) -> List[dict]:
        """Return top-k configurations by score."""
        if self._count == 0:
            return []
        scores = self._buffer[:self._count, self.max_n]
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for i in top_idx:
            row = self._buffer[i]
            results.append({
                'rank': len(results) + 1,
                'score': float(row[self.max_n]),
                'config_idx': int(row[self.max_n + 1 + self.max_n]),
                'eigenvalues': row[:self.max_n][row[:self.max_n] != 0].tolist(),
            })
        return results

    def ram_usage_mb(self) -> float:
        return self._buffer.nbytes / (1024 * 1024)

    def save_checkpoint(self, path: str):
        """Save to compressed numpy format."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez_compressed(path,
                            buffer=self._buffer[:self._count],
                            count=self._count,
                            best_score=self._best_score,
                            best_idx=self._best_idx)

    def load_checkpoint(self, path: str):
        """Load from compressed numpy format."""
        data = np.load(path)
        buf = data['buffer']
        n = len(buf)
        self._buffer[:n] = buf
        self._count = int(data['count'])
        self._best_score = float(data['best_score'])
        self._best_idx = int(data['best_idx'])


# ═══════════════════════════════════════════════════════════
# BATCH SCORER
# ═══════════════════════════════════════════════════════════

def score_batch(G_stack: np.ndarray, n_nodes: np.ndarray,
                score_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """Score a batch of G matrices simultaneously.

    Uses vectorized LAPACK via np.linalg.eigh on the batch.
    Skips near-zero eigenvalues (ground node) before computing
    harmonic ratios.

    Args:
        G_stack: (batch, max_n, max_n) -- padded G matrices
        n_nodes: (batch,) -- actual node count per config
        score_fn: Scoring function(eigvals, cons_score, dom_interval) -> float

    Returns:
        (scores, eigvals_batch) -- (batch,) scores and (batch, max_n) eigenvalues
    """
    batch_size = G_stack.shape[0]
    max_n = G_stack.shape[1]

    # Batch eigendecomposition
    all_eigvals = np.linalg.eigvalsh(G_stack)  # (batch, max_n)

    scores = np.zeros(batch_size, dtype=np.float64)
    for i in range(batch_size):
        n = n_nodes[i]
        # Extract real eigenvalues for this config's actual size
        ev = np.sort(np.abs(all_eigvals[i, -n:]))[::-1]  # top-n by magnitude

        # Skip near-zero eigenvalues for correct ratio computation
        ev_clean = _skip_zero_eigenvalues(ev)

        # Quick consonance + interval on clean eigenvalues
        k = min(len(ev_clean), 10)
        ratios = ev_clean[:k] / ev_clean[0]
        cons = float(consonance_score_from_ratios(ratios))

        if len(ratios) > 1 and abs(ratios[1]) > 1e-30:
            r = abs(ratios[0] / ratios[1])
            dom, _ = nearest_consonant(r)
        else:
            dom = 'unison'

        # Score on clean eigenvalues
        scores[i] = score_fn(ev_clean, cons, dom)

    return scores, all_eigvals


# ═══════════════════════════════════════════════════════════
# MAIN EXPLORER
# ═══════════════════════════════════════════════════════════

class ConfigurationExplorer:
    """Memory-first configuration space explorer.

    Precomputes a grid of MNA matrices, explores the manifold via
    batch-scored perturbations, and stores all results in RAM.
    """

    def __init__(self, config: ExplorerConfig = None):
        self.config = config or ExplorerConfig()
        self.manifold = PrecomputedManifold(max_n=32)
        self.results = ResultsBuffer(
            max_results=self.config.max_results, max_n=32)
        # score_fn override takes priority, then target name lookup
        if self.config.score_fn is not None:
            self.score_fn = self.config.score_fn
        else:
            self.score_fn = _SCORE_FUNCS.get(self.config.target, _score_custom)
        self._step = 0
        self._start_time = 0.0
        self._last_stats_time = 0.0
        self._current_idx = 0  # Current position in manifold
        # Navigation anti-camping state (allocated lazily)
        self._visit_counts = None
        self._recent_best_indices: List[int] = []
        self._max_recent = 10

    def precompute(self, progress: bool = True):
        """Precompute configuration grid in parallel."""
        n = self.config.n_precompute
        n_workers = max(1, _N_CORES - 2)

        def cb(done, total):
            if progress:
                print(f"  {done}/{total} ({100*done//total}%)", flush=True)

        if self.config.scale_ram:
            # RAM-scaled build: use half of ram_limit for manifold
            ram_target = self.config.ram_limit_gb * 0.5
            if progress:
                print(f"Scaling manifold to {ram_target:.1f}GB RAM "
                      f"with {n_workers} workers...")
            t0 = time.time()
            self.manifold.scale_to_ram(ram_target, n_workers=n_workers,
                                       progress_cb=cb if progress else None)
        else:
            if progress:
                print(f"Precomputing {n} configurations with {n_workers} workers...")
            t0 = time.time()
            self.manifold.build(n, n_workers=n_workers,
                                progress_cb=cb if progress else None)

        dt = time.time() - t0
        if progress:
            ram = self.manifold.ram_usage_mb()
            print(f"Precomputed {self.manifold.n_configs:,} configs in {dt:.1f}s "
                  f"({ram:.0f}MB RAM)")

        # Initialize visit tracking
        self._visit_counts = np.zeros(self.manifold.n_configs, dtype=np.int32)

    def _perturb_batch(self, center_idx: int, batch_size: int,
                       rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of perturbations around a center config."""
        n = self.manifold._n_nodes[center_idx]
        G_center = self.manifold._G[center_idx].copy()
        max_n = self.manifold.max_n

        G_stack = np.zeros((batch_size, max_n, max_n), dtype=np.float64)
        n_nodes = np.full(batch_size, n, dtype=np.int32)

        for b in range(batch_size):
            G_pert = G_center.copy()
            noise = rng.standard_normal((max_n, max_n)) * 0.5
            noise = 0.5 * (noise + noise.T)
            np.fill_diagonal(noise, 0)
            G_pert[:n, :n] += noise[:n, :n]
            for i in range(n):
                off_diag = G_pert[i, :n].copy()
                off_diag[i] = 0
                G_pert[i, i] = -off_diag.sum() + 1e-6
            G_stack[b] = G_pert

        return G_stack, n_nodes

    def run_step(self) -> int:
        """Execute one exploration step: score batch_size perturbations.

        Navigation with anti-camping: tracks visit counts per config,
        forces exploration when over-visited, adds exploration bonus.

        Returns number of configs scored.
        """
        batch_size = self.config.batch_size
        rng = np.random.default_rng(self._step * 7919 + 42)

        # Lazy init visit counts if needed
        if self._visit_counts is None:
            self._visit_counts = np.zeros(self.manifold.n_configs, dtype=np.int32)

        # Navigate: anti-camping logic
        visits = self._visit_counts[self._current_idx]

        if visits >= 3:
            # Force jump to least-visited region, avoid recent bests
            candidates = np.where(self._visit_counts < 2)[0]
            if len(candidates) == 0:
                candidates = np.argsort(self._visit_counts)[:100]
            recent_set = set(self._recent_best_indices)
            filtered = [c for c in candidates if c not in recent_set]
            if not filtered:
                filtered = list(candidates[:100] if hasattr(candidates, '__len__') else candidates)
            self._current_idx = int(rng.choice(filtered))
        elif self._step % 5 == 0 and self.results.count > 0:
            # Exploit: perturb around best
            best = self.results.best_configurations(1)
            if best:
                self._current_idx = best[0]['config_idx'] % self.manifold.n_configs
        elif rng.random() < 0.3:
            # Random exploration jump
            self._current_idx = rng.integers(0, self.manifold.n_configs)

        # Track visit
        self._visit_counts[self._current_idx] += 1

        # Generate perturbation batch
        G_stack, n_nodes = self._perturb_batch(self._current_idx, batch_size, rng)

        # Batch score
        scores, eigvals_batch = score_batch(G_stack, n_nodes, self.score_fn)

        # Exploration bonus: penalize over-visited configs
        visit_fraction = self._visit_counts[self._current_idx] / max(self._step + 1, 1)
        exploration_bonus = 0.1 * (1.0 - min(visit_fraction, 1.0))
        scores_with_bonus = scores + exploration_bonus

        # Store results (with bonus so exploration is rewarded)
        config_indices = np.full(batch_size, self._current_idx, dtype=np.int32)
        self.results.add_batch(eigvals_batch, scores_with_bonus, config_indices, self._step)

        # Track recent bests
        best_local = np.argmax(scores)
        if scores[best_local] > self.results.best_score * 0.95:
            self._recent_best_indices.append(self._current_idx)
            if len(self._recent_best_indices) > self._max_recent:
                self._recent_best_indices.pop(0)

        # Move to best neighbor if it improves the manifold
        if scores[best_local] > self.manifold._cons_scores[self._current_idx]:
            n = n_nodes[best_local]
            self.manifold._G[self._current_idx] = G_stack[best_local]
            self.manifold._eigvals[self._current_idx, :] = 0
            self.manifold._eigvals[self._current_idx, :n] = np.sort(
                np.abs(eigvals_batch[best_local, -n:]))[::-1]
            self.manifold._cons_scores[self._current_idx] = scores[best_local]

        self._step += 1
        return batch_size

    def run(self, n_steps: int, progress: bool = True) -> dict:
        """Run exploration loop for n_steps."""
        self._start_time = time.time()
        self._last_stats_time = self._start_time
        total_configs = 0

        for step in range(n_steps):
            n_scored = self.run_step()
            total_configs += n_scored

            if self._step % self.config.checkpoint_interval == 0:
                self._checkpoint()

            now = time.time()
            if progress and (now - self._last_stats_time) >= self.config.stats_interval:
                self._print_stats(total_configs)
                self._last_stats_time = now

        elapsed = time.time() - self._start_time
        rate = total_configs / max(elapsed, 0.001)

        if progress:
            self._print_stats(total_configs, final=True)

        return {
            'steps': n_steps,
            'total_configs_scored': total_configs,
            'elapsed_seconds': round(elapsed, 2),
            'configs_per_second': round(rate, 1),
            'best_score': self.results.best_score,
            'results_stored': self.results.count,
            'manifold_ram_mb': round(self.manifold.ram_usage_mb(), 1),
            'results_ram_mb': round(self.results.ram_usage_mb(), 1),
        }

    def run_forever(self, duration: float = None, progress: bool = True) -> dict:
        """Run exploration loop indefinitely or for duration seconds."""
        if duration is None:
            duration = self.config.duration

        self._start_time = time.time()
        self._last_stats_time = self._start_time
        total_configs = 0
        n_steps = 0

        try:
            while True:
                n_scored = self.run_step()
                total_configs += n_scored
                n_steps += 1

                if self._step % self.config.checkpoint_interval == 0:
                    self._checkpoint()

                now = time.time()
                if progress and (now - self._last_stats_time) >= self.config.stats_interval:
                    self._print_stats(total_configs)
                    self._last_stats_time = now

                if duration is not None and (now - self._start_time) >= duration:
                    break

        except KeyboardInterrupt:
            if progress:
                print("\n  Interrupted by user")

        elapsed = time.time() - self._start_time
        rate = total_configs / max(elapsed, 0.001)

        if progress:
            self._print_stats(total_configs, final=True)

        self._checkpoint()

        return {
            'steps': n_steps,
            'total_configs_scored': total_configs,
            'elapsed_seconds': round(elapsed, 2),
            'configs_per_second': round(rate, 1),
            'best_score': self.results.best_score,
            'results_stored': self.results.count,
            'manifold_ram_mb': round(self.manifold.ram_usage_mb(), 1),
            'results_ram_mb': round(self.results.ram_usage_mb(), 1),
        }

    def _print_stats(self, total_configs: int, final: bool = False):
        """Print current exploration statistics with level info."""
        elapsed = time.time() - self._start_time
        rate = total_configs / max(elapsed, 0.001)
        ram_mb = self.manifold.ram_usage_mb() + self.results.ram_usage_mb()

        # Level info from score_fn if it's an ExplorationTarget
        if isinstance(self.score_fn, ExplorationTarget):
            level_str = f"L{self.score_fn.tensor_level}({self.score_fn.level_name})"
        else:
            level_str = "unknown"

        try:
            import psutil
            proc = psutil.Process()
            rss = proc.memory_info().rss / (1024**3)
            cpu = proc.cpu_percent(interval=0.1)
            ram_str = f"RAM={rss:.1f}GB"
            cpu_str = f"cpu={cpu:.0f}%"
        except ImportError:
            ram_str = f"buf={ram_mb:.0f}MB"
            cpu_str = ""

        above_05 = int(np.sum(self.results.scores() > 0.5)) if self.results.count > 0 else 0

        prefix = "FINAL" if final else f"step={self._step}"
        print(f"  {prefix} level={level_str} rate={rate:.0f} configs/s "
              f"best={self.results.best_score:.4f} "
              f"{ram_str} {cpu_str} stored={self.results.count} above_0.5={above_05}",
              flush=True)

    def _checkpoint(self):
        """Save results to disk."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        path = os.path.join(self.config.log_dir, 'explorer_checkpoint.npz')
        self.results.save_checkpoint(path)

    def best_configurations(self, top_k: int = 10) -> List[dict]:
        """Return top scoring configurations."""
        return self.results.best_configurations(top_k)

    def load_checkpoint(self):
        """Resume from saved checkpoint."""
        path = os.path.join(self.config.log_dir, 'explorer_checkpoint.npz')
        if os.path.exists(path):
            self.results.load_checkpoint(path)
            return True
        return False

    def diagnose(self, n_samples: int = 1000) -> dict:
        """Run diagnostic analysis on the scoring landscape.

        Scores n_samples configs from manifold, prints statistics.
        Returns diagnostic data without running exploration.
        """
        rng = np.random.default_rng(12345)
        scores_list = []
        ratio_patterns = []

        n_to_score = min(n_samples, self.manifold.n_configs)
        for i in range(n_to_score):
            idx = i
            ev = self.manifold.get_eigvals(idx)
            ev_clean = _skip_zero_eigenvalues(np.sort(np.abs(ev))[::-1])

            k = min(len(ev_clean), 10)
            ratios = ev_clean[:k] / ev_clean[0]
            cons = float(consonance_score_from_ratios(ratios))

            if len(ratios) > 1 and abs(ratios[1]) > 1e-30:
                dom, _ = nearest_consonant(abs(ratios[0] / ratios[1]))
            else:
                dom = 'unison'

            s = self.score_fn(ev_clean, cons, dom)
            scores_list.append(s)
            ratio_patterns.append(tuple(np.round(ratios[:4], 3)))

        scores = np.array(scores_list)

        # Score histogram (10 buckets)
        hist, bin_edges = np.histogram(scores, bins=10, range=(0, 1))

        # Top-10 ratio patterns by score
        sorted_idx = np.argsort(-scores)
        top_patterns = [(ratio_patterns[i], float(scores[i]))
                        for i in sorted_idx[:10]]

        # Recommended target ratios (average of top-5)
        if len(sorted_idx) >= 5:
            top_ratios = [ratio_patterns[i] for i in sorted_idx[:5]]
            max_len = max(len(r) for r in top_ratios)
            padded = [list(r) + [0] * (max_len - len(r)) for r in top_ratios]
            recommended = np.mean(padded, axis=0)
        else:
            recommended = np.array([1.0])

        # Estimated time to find score > 0.8
        above_08 = int(np.sum(scores > 0.8))
        if above_08 > 0:
            prob_08 = above_08 / len(scores)
            est_configs = int(1.0 / prob_08)
            est_time = est_configs / (self.config.batch_size * 50)
        else:
            est_configs = -1
            est_time = float('inf')

        # Print report
        print(f"\n{'='*60}")
        print(f"  DIAGNOSTIC REPORT ({n_to_score} samples)")
        print(f"{'='*60}")
        print(f"\n  Score Histogram:")
        max_count = max(max(hist), 1)
        for i in range(len(hist)):
            bar = '#' * (hist[i] * 40 // max_count)
            print(f"    [{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}] {hist[i]:5d} {bar}")

        print(f"\n  Top-10 Eigenvalue Ratio Patterns:")
        for rank, (pattern, sc) in enumerate(top_patterns, 1):
            print(f"    {rank:2d}. score={sc:.4f}  ratios={list(pattern)}")

        print(f"\n  Recommended target_ratios: {list(np.round(recommended, 4))}")

        if est_time < float('inf'):
            print(f"\n  Estimated time to score>0.8: ~{est_time:.0f}s "
                  f"({est_configs} configs, {above_08}/{n_to_score} above 0.8 in sample)")
        else:
            print(f"\n  Estimated time to score>0.8: UNLIKELY "
                  f"(0/{n_to_score} above 0.8 in sample)")
            print(f"  Consider adjusting target_ratios or scoring weights.")

        return {
            'histogram': (hist.tolist(), bin_edges.tolist()),
            'top_patterns': top_patterns,
            'recommended_ratios': recommended.tolist(),
            'estimated_time_to_08': est_time,
            'mean_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'std_score': float(np.std(scores)),
        }
