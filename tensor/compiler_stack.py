"""Cross-language compilation mapping via coarse-graining.

φ (coarse-graining) IS the compiler transformation.
Each compilation step preserves eigenvalue ratios (computational semantics)
while reducing abstraction. φ⁻¹ (lifting) IS hardware-aware optimization.

Language levels map to tensor levels:
  L_python:   AST nodes, function calls, classes (L2)
  L_bytecode: Python bytecode instructions
  L_llvm:     LLVM IR (stub)
  L_asm:      x86/ARM assembly instructions
  L_hardware: Logic gates / transistors (L3, ECEMath domain)
"""
import ast
import dis
import os
import sys
import json
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.matrix import MNASystem
from core.coarsening import CoarseGrainingOperator, CoarseGrainResult
from core.sparse_solver import compute_harmonic_signature
from tensor.code_graph import CodeGraph


# x86 instruction latency table (simplified, in cycles)
X86_LATENCIES = {
    'mov': 1, 'add': 1, 'sub': 1, 'imul': 3, 'idiv': 20,
    'and': 1, 'or': 1, 'xor': 1, 'shl': 1, 'shr': 1,
    'cmp': 1, 'test': 1, 'jmp': 1, 'je': 1, 'jne': 1,
    'call': 3, 'ret': 1, 'push': 1, 'pop': 1,
    'lea': 1, 'nop': 1,
    'addss': 3, 'mulss': 5, 'divss': 14,  # SSE scalar
    'vaddps': 3, 'vmulps': 5, 'vdivps': 14,  # AVX vector
}

# Bytecode opcode throughput (relative, higher = faster)
OPCODE_THROUGHPUT = {
    'LOAD_FAST': 1.0, 'STORE_FAST': 1.0,
    'LOAD_CONST': 0.9, 'LOAD_GLOBAL': 0.5,
    'LOAD_ATTR': 0.4, 'STORE_ATTR': 0.4,
    'BINARY_ADD': 0.8, 'BINARY_SUBTRACT': 0.8,
    'BINARY_MULTIPLY': 0.7, 'BINARY_TRUE_DIVIDE': 0.5,
    'COMPARE_OP': 0.8, 'POP_JUMP_IF_FALSE': 0.9,
    'POP_JUMP_IF_TRUE': 0.9, 'JUMP_FORWARD': 1.0,
    'JUMP_ABSOLUTE': 1.0, 'CALL_FUNCTION': 0.3,
    'RETURN_VALUE': 0.9, 'POP_TOP': 1.0,
    'BUILD_TUPLE': 0.6, 'BUILD_LIST': 0.5,
    'UNPACK_SEQUENCE': 0.7, 'FOR_ITER': 0.6,
    'GET_ITER': 0.7,
}


@dataclass
class PhiResult:
    """Result of applying φ between language levels."""
    projection: np.ndarray
    eigenvalue_ratios_high: np.ndarray
    eigenvalue_ratios_low: np.ndarray
    ratio_error: float
    level_high: str
    level_low: str


LEVELS = {
    'python': 5,
    'bytecode': 4,
    'llvm': 3,
    'asm': 2,
    'hardware': 1,
}


class CompilerStack:
    """Maps between language levels using φ.

    Each level's MNA matrix is a coarse-graining of the level below it.
    The eigenvalue ratios that survive ALL levels of coarsening are the
    computationally fundamental operations.
    """

    def __init__(self, hardware_profile_path: Optional[str] = None):
        self._mna_cache: Dict[str, MNASystem] = {}
        self._hardware_profile = None
        if hardware_profile_path and os.path.exists(hardware_profile_path):
            with open(hardware_profile_path) as f:
                self._hardware_profile = json.load(f)

    def python_to_mna(self, source_path: str) -> MNASystem:
        """Wrap code_graph.py's existing Python→MNA conversion."""
        if os.path.isdir(source_path):
            cg = CodeGraph.from_directory(source_path, max_files=200)
        else:
            cg = CodeGraph.from_directory(os.path.dirname(source_path), max_files=200)
        mna = cg.to_mna()
        self._mna_cache['python'] = mna
        return mna

    def bytecode_to_mna(self, source_path: str) -> MNASystem:
        """Extract Python bytecode and build MNA.

        Nodes = unique opcodes
        Edges = opcode→opcode transitions (frequency-weighted)
        C matrix = stack depth at each opcode
        G matrix = opcode throughput
        """
        with open(source_path, 'r', errors='replace') as f:
            source = f.read()

        code = compile(source, source_path, 'exec')
        instructions = list(dis.get_instructions(code))

        # Also get instructions from nested code objects
        for const in code.co_consts:
            if hasattr(const, 'co_code'):
                try:
                    instructions.extend(dis.get_instructions(const))
                except Exception:
                    pass

        if not instructions:
            return self._empty_mna()

        # Collect unique opcodes and transitions
        opcode_names = list(set(instr.opname for instr in instructions))
        opcode_names.sort()
        n = len(opcode_names)
        if n == 0:
            return self._empty_mna()

        opcode_idx = {name: i for i, name in enumerate(opcode_names)}

        # Count transitions
        transitions = Counter()
        stack_depths = Counter()
        opcode_counts = Counter()

        depth = 0
        for i, instr in enumerate(instructions):
            opcode_counts[instr.opname] += 1
            stack_depths[instr.opname] = max(stack_depths.get(instr.opname, 0), depth)

            # Estimate stack effect
            if instr.opname.startswith('LOAD'):
                depth += 1
            elif instr.opname.startswith('STORE') or instr.opname == 'POP_TOP':
                depth = max(0, depth - 1)
            elif instr.opname.startswith('BINARY'):
                depth = max(0, depth - 1)

            if i + 1 < len(instructions):
                transitions[(instr.opname, instructions[i + 1].opname)] += 1

        # Build G matrix (conductance = transition frequency * throughput)
        G = np.zeros((n, n))
        for (src, dst), count in transitions.items():
            i, j = opcode_idx[src], opcode_idx[dst]
            if i == j:
                continue
            throughput_src = OPCODE_THROUGHPUT.get(src, 0.5)
            throughput_dst = OPCODE_THROUGHPUT.get(dst, 0.5)
            g = count * (throughput_src + throughput_dst) / 2.0
            G[i, i] += g
            G[j, j] += g
            G[i, j] -= g
            G[j, i] -= g

        # Build C matrix (stack depth as capacitance)
        C = np.zeros((n, n))
        for name, idx in opcode_idx.items():
            depth_val = max(1, stack_depths.get(name, 1))
            C[idx, idx] = float(depth_val)

        # Diagonal loading for stability
        G += 1e-6 * np.eye(n)
        C += 1e-6 * np.eye(n)

        node_map = {i: i for i in range(n)}
        mna = MNASystem(
            C=C, G=G, n_nodes=n, n_branches=0, n_total=n,
            node_map=node_map, branch_map={}, branch_info=[],
        )
        self._mna_cache['bytecode'] = mna
        return mna

    def asm_to_mna(self, binary_path: Optional[str] = None) -> MNASystem:
        """Build MNA from x86 instruction model.

        Uses the x86 latency table as a stub. Real implementation would
        use objdump or pyelftools to disassemble an actual binary.
        """
        # Stub: build from x86 latency table
        instr_names = list(X86_LATENCIES.keys())
        n = len(instr_names)
        instr_idx = {name: i for i, name in enumerate(instr_names)}

        # G matrix: instruction latency as resistance (1/latency = conductance)
        G = np.zeros((n, n))
        # Connect related instructions (ALU ops, memory ops, branch ops)
        alu_ops = ['add', 'sub', 'imul', 'idiv', 'and', 'or', 'xor', 'shl', 'shr']
        mem_ops = ['mov', 'push', 'pop', 'lea']
        branch_ops = ['jmp', 'je', 'jne', 'call', 'ret']
        simd_ops = ['addss', 'mulss', 'divss', 'vaddps', 'vmulps', 'vdivps']

        for group in [alu_ops, mem_ops, branch_ops, simd_ops]:
            for a in group:
                for b in group:
                    if a != b and a in instr_idx and b in instr_idx:
                        i, j = instr_idx[a], instr_idx[b]
                        lat_a = X86_LATENCIES[a]
                        lat_b = X86_LATENCIES[b]
                        g = 1.0 / (lat_a + lat_b)
                        G[i, i] += g
                        G[j, j] += g
                        G[i, j] -= g
                        G[j, i] -= g

        # C matrix: pipeline depth
        C = np.zeros((n, n))
        for name, idx in instr_idx.items():
            C[idx, idx] = float(X86_LATENCIES[name])

        G += 1e-6 * np.eye(n)
        C += 1e-6 * np.eye(n)

        node_map = {i: i for i in range(n)}
        mna = MNASystem(
            C=C, G=G, n_nodes=n, n_branches=0, n_total=n,
            node_map=node_map, branch_map={}, branch_info=[],
        )
        self._mna_cache['asm'] = mna
        return mna

    def phi_between(self, level_high: str, level_low: str,
                    mna_high: MNASystem) -> PhiResult:
        """Apply φ to map from high abstraction to low.

        The projection matrix P tells you which high-level constructs
        map to which low-level primitives.
        """
        n = mna_high.n_total
        if n < 3:
            # Too small to coarsen
            return PhiResult(
                projection=np.eye(n),
                eigenvalue_ratios_high=np.array([1.0]),
                eigenvalue_ratios_low=np.array([1.0]),
                ratio_error=0.0,
                level_high=level_high,
                level_low=level_low,
            )

        k = max(1, min(n - 1, int(np.sqrt(n))))
        try:
            phi = CoarseGrainingOperator(mna_high, k=k, tolerance=1.0)
            result = phi.coarsen()

            return PhiResult(
                projection=result.projection,
                eigenvalue_ratios_high=result.eigenvalue_ratios_fine,
                eigenvalue_ratios_low=result.eigenvalue_ratios_coarse,
                ratio_error=result.ratio_error,
                level_high=level_high,
                level_low=level_low,
            )
        except ValueError:
            return PhiResult(
                projection=np.eye(min(k, n)),
                eigenvalue_ratios_high=np.array([1.0]),
                eigenvalue_ratios_low=np.array([1.0]),
                ratio_error=0.0,
                level_high=level_high,
                level_low=level_low,
            )

    def optimal_high_level(self, hardware_mna: MNASystem,
                           target_behavior: str) -> dict:
        """Reverse compilation — hardware-aware optimizer.

        Given the hardware's MNA eigenstructure, recommend high-level code
        structure that compiles to hardware-efficient code.
        """
        # Get hardware eigenstructure
        hw_eigvals = np.sort(np.abs(np.linalg.eigvalsh(hardware_mna.G)))[::-1]
        hw_sig = compute_harmonic_signature(hardware_mna.G,
                                             k=min(hardware_mna.n_total, 10))

        # Recommended structure based on hardware eigenvectors
        n_files = max(2, int(np.sqrt(hardware_mna.n_total)))

        recommendation = {
            'target_behavior': target_behavior,
            'recommended_n_files': n_files,
            'hardware_dominant_interval': hw_sig.dominant_interval,
            'hardware_consonance': hw_sig.consonance_score,
            'recommended_structure': f'{n_files} files with balanced coupling',
            'optimal_coupling': float(1.0 / n_files),
            'hardware_eigvals_top5': hw_eigvals[:5].tolist(),
        }
        return recommendation

    def cross_language_report(self, source_path: str) -> dict:
        """Full analysis of one Python file across all levels."""
        python_mna = self.python_to_mna(source_path)
        bytecode_mna = self.bytecode_to_mna(source_path)
        asm_mna = self.asm_to_mna()  # stub

        phi_py_bc = self.phi_between('python', 'bytecode', python_mna)
        phi_bc_asm = self.phi_between('bytecode', 'asm', bytecode_mna)

        report = {
            'python_mna_nodes': python_mna.n_total,
            'bytecode_mna_nodes': bytecode_mna.n_total,
            'asm_mna_nodes': asm_mna.n_total,
            'phi_python_bytecode': {
                'ratio_error': phi_py_bc.ratio_error,
                'projection_shape': list(phi_py_bc.projection.shape),
            },
            'phi_bytecode_asm': {
                'ratio_error': phi_bc_asm.ratio_error,
                'projection_shape': list(phi_bc_asm.projection.shape),
            },
            'eigenvalue_preservation': {
                'python_to_bytecode': phi_py_bc.ratio_error,
                'bytecode_to_asm': phi_bc_asm.ratio_error,
            },
            'bottleneck_level': (
                'python_to_bytecode' if phi_py_bc.ratio_error > phi_bc_asm.ratio_error
                else 'bytecode_to_asm'
            ),
        }
        return report

    def _empty_mna(self) -> MNASystem:
        return MNASystem(
            C=np.zeros((1, 1)), G=np.zeros((1, 1)),
            n_nodes=1, n_branches=0, n_total=1,
            node_map={0: 0}, branch_map={}, branch_info=[],
        )
