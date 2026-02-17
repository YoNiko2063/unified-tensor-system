"""Hardware profiler: reads THIS machine's capabilities as an MNA matrix.

Populates tensor L3 with real hardware data. The L3 MNA represents the
machine's natural computational geometry — which operations are consonant
with the hardware's eigenstructure.
"""
import os
import re
import sys
import json
import subprocess
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.matrix import MNASystem
from core.sparse_solver import compute_harmonic_signature


@dataclass
class HardwareProfile:
    """Profile of this machine's hardware capabilities."""
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_model: str = ''
    cpu_freq_mhz: float = 0.0
    l1_cache_kb: float = 0
    l2_cache_kb: float = 0
    l3_cache_kb: float = 0
    ram_total_gb: float = 0.0
    simd_capabilities: List[str] = field(default_factory=list)
    numa_nodes: int = 1
    architecture: str = 'x86_64'


# Hardware functional unit nodes for MNA
FUNCTIONAL_UNITS = [
    'ALU', 'FPU', 'SIMD_unit', 'L1_cache',
    'L2_cache', 'L3_cache', 'RAM_bus', 'branch_predictor',
]

# Typical latencies between units (nanoseconds) — used as resistance
# Lower latency = higher conductance
DEFAULT_LATENCIES = {
    ('ALU', 'L1_cache'): 1.0,        # register → L1
    ('FPU', 'L1_cache'): 2.0,        # FP → L1
    ('SIMD_unit', 'L1_cache'): 1.5,  # SIMD → L1
    ('L1_cache', 'L2_cache'): 4.0,   # L1 miss
    ('L2_cache', 'L3_cache'): 12.0,  # L2 miss
    ('L3_cache', 'RAM_bus'): 50.0,   # L3 miss → DRAM
    ('ALU', 'branch_predictor'): 1.0, # branch resolution
    ('ALU', 'FPU'): 2.0,             # int → float
    ('ALU', 'SIMD_unit'): 2.0,       # scalar → vector
    ('FPU', 'SIMD_unit'): 1.0,       # FP → SIMD
}

# Default throughput (GB/s) for functional units
DEFAULT_THROUGHPUT = {
    'ALU': 100.0,
    'FPU': 50.0,
    'SIMD_unit': 200.0,
    'L1_cache': 500.0,
    'L2_cache': 200.0,
    'L3_cache': 100.0,
    'RAM_bus': 50.0,
    'branch_predictor': 100.0,
}

# Default capacities (in normalized units)
DEFAULT_CAPACITY = {
    'ALU': 1.0,
    'FPU': 1.0,
    'SIMD_unit': 1.0,
    'L1_cache': 32.0,     # KB (normalized)
    'L2_cache': 256.0,
    'L3_cache': 8192.0,
    'RAM_bus': 16384.0,    # MB
    'branch_predictor': 1.0,
}


class HardwareProfiler:
    """Reads THIS machine's hardware capabilities and expresses them as MNA."""

    def __init__(self):
        self._profile: Optional[HardwareProfile] = None
        self._mna: Optional[MNASystem] = None

    def profile(self) -> HardwareProfile:
        """Read hardware specs from the running system."""
        hp = HardwareProfile()

        # CPU topology via lscpu
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                hp = self._parse_lscpu(result.stdout, hp)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        hp.ram_total_gb = kb / (1024 * 1024)
                        break
        except (FileNotFoundError, ValueError):
            pass

        # SIMD capabilities from /proc/cpuinfo
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
            flags_match = re.search(r'flags\s*:\s*(.+)', content)
            if flags_match:
                flags = flags_match.group(1).split()
                simd_flags = [f for f in flags
                              if f in ('sse', 'sse2', 'sse3', 'ssse3', 'sse4_1',
                                      'sse4_2', 'avx', 'avx2', 'avx512f', 'avx512bw',
                                      'avx512vl', 'avx512dq')]
                hp.simd_capabilities = sorted(set(simd_flags))
        except FileNotFoundError:
            pass

        # Cache sizes from sysfs
        for level, attr in [(1, 'l1_cache_kb'), (2, 'l2_cache_kb'), (3, 'l3_cache_kb')]:
            try:
                cache_dir = f'/sys/devices/system/cpu/cpu0/cache/index{level}'
                if os.path.exists(cache_dir):
                    with open(os.path.join(cache_dir, 'size'), 'r') as f:
                        size_str = f.read().strip()
                    # Parse "32K" or "8192K" or "16M"
                    if size_str.endswith('K'):
                        setattr(hp, attr, float(size_str[:-1]))
                    elif size_str.endswith('M'):
                        setattr(hp, attr, float(size_str[:-1]) * 1024)
            except (FileNotFoundError, ValueError):
                pass

        # CPU frequency
        try:
            freq_path = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'
            if os.path.exists(freq_path):
                with open(freq_path, 'r') as f:
                    hp.cpu_freq_mhz = float(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            pass

        # NUMA topology
        try:
            result = subprocess.run(['numactl', '--hardware'], capture_output=True,
                                    text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'available' in line.lower():
                        parts = line.split()
                        for p in parts:
                            if p.isdigit():
                                hp.numa_nodes = int(p)
                                break
                        break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallbacks
        if hp.cpu_cores == 0:
            hp.cpu_cores = os.cpu_count() or 1
        if hp.cpu_threads == 0:
            hp.cpu_threads = hp.cpu_cores

        self._profile = hp
        return hp

    def _parse_lscpu(self, output: str, hp: HardwareProfile) -> HardwareProfile:
        """Parse lscpu output into HardwareProfile."""
        for line in output.split('\n'):
            line = line.strip()
            if ':' not in line:
                continue
            key, _, val = line.partition(':')
            key = key.strip().lower()
            val = val.strip()

            if key == 'cpu(s)':
                try:
                    hp.cpu_threads = int(val)
                except ValueError:
                    pass
            elif 'core(s) per socket' in key:
                try:
                    hp.cpu_cores = int(val)
                except ValueError:
                    pass
            elif 'model name' in key:
                hp.cpu_model = val
            elif key == 'architecture':
                hp.architecture = val
            elif 'socket(s)' in key:
                try:
                    sockets = int(val)
                    if hp.cpu_cores > 0:
                        hp.cpu_cores *= sockets
                except ValueError:
                    pass
        return hp

    def to_mna(self, profile: Optional[HardwareProfile] = None) -> MNASystem:
        """Convert hardware profile to L3 MNA matrix.

        Nodes = functional units (ALU, FPU, SIMD, caches, RAM, branch predictor)
        Edges = data flow between units
        C matrix = capacity (cache sizes)
        G matrix = throughput (bandwidth as conductance)
        """
        if profile is None:
            profile = self._profile or self.profile()

        n = len(FUNCTIONAL_UNITS)
        unit_idx = {name: i for i, name in enumerate(FUNCTIONAL_UNITS)}

        G = np.zeros((n, n))
        C = np.zeros((n, n))

        # Build G from latencies (conductance = 1/latency, scaled by cores)
        core_scale = max(1, profile.cpu_cores)
        for (src, dst), latency in DEFAULT_LATENCIES.items():
            i, j = unit_idx[src], unit_idx[dst]
            g = core_scale / latency
            G[i, i] += g
            G[j, j] += g
            G[i, j] -= g
            G[j, i] -= g

        # SIMD boost: if AVX/AVX2/AVX512 present, increase SIMD conductance
        simd_boost = 1.0
        if 'avx512f' in profile.simd_capabilities:
            simd_boost = 16.0
        elif 'avx2' in profile.simd_capabilities:
            simd_boost = 8.0
        elif 'avx' in profile.simd_capabilities:
            simd_boost = 4.0
        elif 'sse4_2' in profile.simd_capabilities:
            simd_boost = 2.0

        simd_idx = unit_idx['SIMD_unit']
        G[simd_idx, simd_idx] *= simd_boost

        # Build C from capacity
        for name, idx in unit_idx.items():
            capacity = DEFAULT_CAPACITY.get(name, 1.0)
            # Scale cache capacities by actual values if available
            if name == 'L1_cache' and profile.l1_cache_kb > 0:
                capacity = profile.l1_cache_kb
            elif name == 'L2_cache' and profile.l2_cache_kb > 0:
                capacity = profile.l2_cache_kb
            elif name == 'L3_cache' and profile.l3_cache_kb > 0:
                capacity = profile.l3_cache_kb
            elif name == 'RAM_bus' and profile.ram_total_gb > 0:
                capacity = profile.ram_total_gb * 1024  # GB → MB
            C[idx, idx] = capacity

        # Diagonal loading
        G += 1e-6 * np.eye(n)
        C += 1e-6 * np.eye(n)

        node_map = {i: i for i in range(n)}
        mna = MNASystem(
            C=C, G=G, n_nodes=n, n_branches=0, n_total=n,
            node_map=node_map, branch_map={}, branch_info=[],
        )
        self._mna = mna
        return mna

    def optimal_operation_sequence(self,
                                    target_computation: str) -> List[str]:
        """Given a computation description, recommend optimal operation sequence."""
        if self._mna is None:
            self.to_mna()

        sig = compute_harmonic_signature(self._mna.G, k=min(8, self._mna.n_total))

        # Map dominant interval to operation recommendation
        recommendations = {
            'octave': ['Use SIMD vectorization', 'Batch operations in cache-line units'],
            'fifth': ['Balance ALU and memory operations', 'Use prefetch hints'],
            'fourth': ['Optimize branch prediction', 'Reduce branch mispredicts'],
            'major_third': ['Use FPU pipeline fully', 'Avoid int/float conversions'],
            'minor_third': ['Mixed precision operations', 'Consider GPU offload'],
            'tritone': ['Pipeline stall risk — restructure loops'],
        }

        ops = recommendations.get(sig.dominant_interval,
                                   ['General optimization: reduce cache misses'])

        # Add SIMD-specific advice
        profile = self._profile or HardwareProfile()
        if 'avx512f' in profile.simd_capabilities:
            ops.append('AVX-512 available: use 512-bit vector operations')
        elif 'avx2' in profile.simd_capabilities:
            ops.append('AVX2 available: use 256-bit vector operations')

        ops.insert(0, f'Target: {target_computation}')
        return ops

    def hardware_report(self) -> str:
        """Markdown report of this machine's computational geometry."""
        profile = self._profile or self.profile()
        mna = self._mna or self.to_mna(profile)

        sig = compute_harmonic_signature(mna.G, k=min(8, mna.n_total))
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(mna.G)))[::-1]

        lines = [
            '# Hardware Computational Geometry Report',
            '',
            '## Machine Profile',
            f'- CPU: {profile.cpu_model or "unknown"}',
            f'- Cores: {profile.cpu_cores}, Threads: {profile.cpu_threads}',
            f'- Frequency: {profile.cpu_freq_mhz:.0f} MHz',
            f'- RAM: {profile.ram_total_gb:.1f} GB',
            f'- L1/L2/L3 Cache: {profile.l1_cache_kb:.0f}K / '
            f'{profile.l2_cache_kb:.0f}K / {profile.l3_cache_kb:.0f}K',
            f'- SIMD: {", ".join(profile.simd_capabilities) or "none detected"}',
            f'- NUMA nodes: {profile.numa_nodes}',
            '',
            '## MNA Eigenstructure',
            f'- Functional units: {len(FUNCTIONAL_UNITS)}',
            f'- Dominant interval: {sig.dominant_interval}',
            f'- Consonance score: {sig.consonance_score:.4f}',
            f'- Stability verdict: {sig.stability_verdict}',
            f'- Top eigenvalues: {eigvals[:5].tolist()}',
            '',
            '## Hardware-Consonant Operations',
        ]

        ops = self.optimal_operation_sequence('general computation')
        for op in ops[1:]:  # Skip target line
            lines.append(f'- {op}')

        lines.append('')
        return '\n'.join(lines)
