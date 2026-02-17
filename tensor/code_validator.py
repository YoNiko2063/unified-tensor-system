"""Code validator: validates generated code against the tensor before committing.

Checks file structure rules, L2 compatibility, hardware resonance (L3),
and cross-level consistency.
"""
import ast
import os
import sys
import tempfile
import time
import numpy as np
from dataclasses import dataclass
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
from tensor.core import UnifiedTensor
from tensor.code_graph import CodeGraph


@dataclass
class ValidationResult:
    """Result of validating proposed code against the tensor."""
    approved: bool
    consonance_before: float
    consonance_after: float
    consonance_delta: float
    resonance_before: float
    resonance_after: float
    resonance_delta: float
    reason: str
    suggestions: List[str]


class CodeValidator:
    """Validates generated code against tensor L2 and L3."""

    def __init__(self, tensor: UnifiedTensor, codebase_root: str,
                 max_files: int = 200,
                 consonance_threshold: float = -0.1,
                 resonance_threshold: float = -0.1):
        self.tensor = tensor
        self.codebase_root = os.path.abspath(codebase_root)
        self.max_files = max_files
        self.consonance_threshold = consonance_threshold
        self.resonance_threshold = resonance_threshold

    def _current_metrics(self) -> tuple:
        """Get current L2 consonance and L2-L3 resonance."""
        sig = self.tensor.harmonic_signature(2)
        consonance = sig.consonance_score

        # L2-L3 resonance (if L3 is populated)
        mna3 = self.tensor._mna.get(3)
        if mna3 is not None:
            resonance = self.tensor.cross_level_resonance(2, 3)
        else:
            resonance = 0.5  # neutral if no hardware data

        return consonance, resonance

    def validate(self, new_file_path: str,
                 proposed_content: str) -> ValidationResult:
        """Validate proposed code before writing.

        1. Parse proposed_content as AST
        2. Build hypothetical new CodeGraph with this file
        3. Compute new L2 MNA
        4. Measure consonance and resonance changes
        5. Approve or reject
        """
        suggestions = []

        # 1. Parse AST
        try:
            tree = ast.parse(proposed_content)
        except SyntaxError as e:
            return ValidationResult(
                approved=False,
                consonance_before=0, consonance_after=0, consonance_delta=0,
                resonance_before=0, resonance_after=0, resonance_delta=0,
                reason=f'Syntax error: {e}',
                suggestions=['Fix syntax errors before submitting'],
            )

        # Get current metrics
        cons_before, res_before = self._current_metrics()

        # 2. Write proposed file temporarily, re-parse codebase
        abs_path = os.path.abspath(new_file_path)
        existed = os.path.exists(abs_path)
        old_content = None
        if existed:
            with open(abs_path, 'r', errors='replace') as f:
                old_content = f.read()

        try:
            # Write proposed content
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, 'w') as f:
                f.write(proposed_content)

            # Re-parse codebase
            cg = CodeGraph.from_directory(self.codebase_root, max_files=self.max_files)
            mna = cg.to_mna()
            self.tensor.update_level(2, mna, t=time.time())

            # Measure new metrics
            cons_after, res_after = self._current_metrics()

        finally:
            # Restore original state
            if existed and old_content is not None:
                with open(abs_path, 'w') as f:
                    f.write(old_content)
                # Re-parse to restore tensor state
                cg_restore = CodeGraph.from_directory(self.codebase_root,
                                                      max_files=self.max_files)
                mna_restore = cg_restore.to_mna()
                self.tensor.update_level(2, mna_restore, t=time.time())
            elif not existed and os.path.exists(abs_path):
                os.remove(abs_path)
                cg_restore = CodeGraph.from_directory(self.codebase_root,
                                                      max_files=self.max_files)
                mna_restore = cg_restore.to_mna()
                self.tensor.update_level(2, mna_restore, t=time.time())

        cons_delta = cons_after - cons_before
        res_delta = res_after - res_before

        # 5. Check line count
        n_lines = proposed_content.count('\n') + 1
        if n_lines > 300:
            suggestions.append(f'File has {n_lines} lines — consider splitting (target <200)')

        # 6. Decision
        approved = True
        reasons = []

        if cons_delta < self.consonance_threshold:
            approved = False
            reasons.append(
                f'Consonance degraded by {cons_delta:.4f} '
                f'(threshold: {self.consonance_threshold})')
            suggestions.append('Reduce coupling: fewer cross-module dependencies')

        if res_delta < self.resonance_threshold:
            approved = False
            reasons.append(
                f'Hardware resonance degraded by {res_delta:.4f} '
                f'(threshold: {self.resonance_threshold})')
            suggestions.append('Restructure for better hardware alignment')

        if approved:
            reason = 'Approved: consonance and resonance maintained or improved'
        else:
            reason = '; '.join(reasons)

        return ValidationResult(
            approved=approved,
            consonance_before=cons_before,
            consonance_after=cons_after,
            consonance_delta=cons_delta,
            resonance_before=res_before,
            resonance_after=res_after,
            resonance_delta=res_delta,
            reason=reason,
            suggestions=suggestions,
        )

    def suggest_structure(self, behavior_description: str) -> dict:
        """Suggest file structure that fits the existing codebase.

        Maximizes consonance with current L2 MNA and resonance with L3.
        """
        # Current L2 analysis
        sig = self.tensor.harmonic_signature(2)
        mna2 = self.tensor._mna.get(2)

        n_existing = mna2.n_total if mna2 else 0

        # Estimate optimal file count from current eigenstructure
        if mna2 is not None and mna2.n_total > 1:
            eigvals = np.sort(np.abs(np.linalg.eigvalsh(mna2.G)))[::-1]
            # Number of significant eigenvalues = natural module count
            threshold = eigvals[0] * 0.01
            n_significant = int(np.sum(eigvals > threshold))
        else:
            n_significant = 3

        # L3 influence
        mna3 = self.tensor._mna.get(3)
        hw_recommendation = ''
        if mna3 is not None:
            hw_sig = compute_harmonic_signature(mna3.G, k=min(8, mna3.n_total))
            resonance = self.tensor.cross_level_resonance(2, 3)
            hw_recommendation = (
                f'Hardware interval: {hw_sig.dominant_interval}, '
                f'resonance: {resonance:.4f}')

        suggestion = {
            'behavior': behavior_description,
            'recommended_files': max(1, min(5, n_significant // 3 + 1)),
            'max_lines_per_file': 200,
            'target_consonance': sig.consonance_score,
            'current_modules': n_existing,
            'dominant_interval': sig.dominant_interval,
            'hardware_recommendation': hw_recommendation,
            'structure_advice': [
                'One file per responsibility',
                f'Keep files under 200 lines',
                f'Match existing coupling pattern ({sig.dominant_interval})',
                'Minimize circular dependencies',
            ],
        }
        return suggestion
