"""BorrowPredictor — wraps code_gen_experiment.py classifier for three prediction modes.

Modes:
  from_intent:      Pre-gate — predict from IntentSpec's expected BV
  from_template:    Design-time — predict from template's design BV
  validate_generated: Post-gate — AST-extract BV from generated Rust, predict + try compile
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Import from code_gen_experiment
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from optimization.code_gen_experiment import (
    WEIGHTS,
    D_SEP,
    e_borrow,
    feature_vec,
    load_classifier,
    predict,
    extract_borrow_vector,
    try_compile,
    METRICS_JSONL,
)


@dataclass
class PredictionResult:
    """Result of a borrow prediction."""
    borrow_vector: Tuple[float, ...]
    e_borrow: float
    predicted_compile: bool
    probability: float
    actual_compile: Optional[bool] = None
    compile_stderr: str = ""
    source: str = ""  # "intent", "template", "ast"


class BorrowPredictor:
    """Three-mode borrow prediction wrapping code_gen_experiment.py."""

    def __init__(self, metrics_path: str = METRICS_JSONL) -> None:
        self._metrics_path = metrics_path
        self._clf = None
        self._scaler = None

    def _ensure_classifier(self) -> None:
        if self._clf is None:
            self._clf, self._scaler = load_classifier(self._metrics_path)

    def from_intent(self, intent) -> PredictionResult:
        """Pre-gate: predict from IntentSpec's expected BorrowVector."""
        self._ensure_classifier()
        bv = intent.expected_borrow_vector()
        eb = e_borrow(bv)
        pred, prob = predict(self._clf, self._scaler, bv)
        return PredictionResult(
            borrow_vector=bv,
            e_borrow=eb,
            predicted_compile=pred,
            probability=prob,
            source="intent",
        )

    def from_template(self, template) -> PredictionResult:
        """Design-time: predict from template's design BorrowVector."""
        self._ensure_classifier()
        bv = template.design_bv
        eb = e_borrow(bv)
        pred, prob = predict(self._clf, self._scaler, bv)
        return PredictionResult(
            borrow_vector=bv,
            e_borrow=eb,
            predicted_compile=pred,
            probability=prob,
            source="template",
        )

    def validate_generated(self, rust_code: str) -> PredictionResult:
        """Post-gate: AST-extract BV from generated Rust, predict + try compile.

        If the borrow_extractor binary is unavailable, falls back to compile-only.
        """
        self._ensure_classifier()

        # AST extraction
        ast_bv = extract_borrow_vector(rust_code)
        if ast_bv is not None:
            bv = ast_bv
            source = "ast"
        else:
            # Fallback: zero BV (conservative — will predict compile=True)
            bv = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            source = "ast_fallback"

        eb = e_borrow(bv)
        pred, prob = predict(self._clf, self._scaler, bv)

        # Actual compilation
        actual, stderr = try_compile(rust_code)

        return PredictionResult(
            borrow_vector=bv,
            e_borrow=eb,
            predicted_compile=pred,
            probability=prob,
            actual_compile=actual,
            compile_stderr=stderr,
            source=source,
        )

    @property
    def d_sep(self) -> float:
        """Compile failure boundary."""
        return D_SEP
