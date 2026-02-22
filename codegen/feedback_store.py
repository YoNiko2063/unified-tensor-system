"""FeedbackStore — JSONL storage for (BV, compile_result) feedback samples.

Collects feedback from pipeline runs and supports retraining the BorrowPredictor
by merging the original 250 metrics.jsonl samples with new feedback.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from optimization.code_gen_experiment import WEIGHTS, METRICS_JSONL


@dataclass
class FeedbackSample:
    """Single feedback record from pipeline execution."""
    b1: float
    b2: float
    b3: float
    b4: float
    b5: float
    b6: float
    compile_success: bool
    template_name: str = ""
    source: str = ""   # "ast", "template", "intent"
    timestamp: str = ""


_DEFAULT_FEEDBACK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "feedback.jsonl",
)


class FeedbackStore:
    """Append-only JSONL feedback store with retrain support."""

    def __init__(self, path: str = _DEFAULT_FEEDBACK_PATH) -> None:
        self._path = path

    def record(self, sample: FeedbackSample) -> None:
        """Append a feedback sample to the JSONL file."""
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(sample)) + "\n")

    def record_from_result(self, prediction_result, template_name: str = "") -> None:
        """Record feedback from a PredictionResult (from borrow_predictor)."""
        if prediction_result.actual_compile is None:
            return
        import time
        sample = FeedbackSample(
            b1=prediction_result.borrow_vector[0],
            b2=prediction_result.borrow_vector[1],
            b3=prediction_result.borrow_vector[2],
            b4=prediction_result.borrow_vector[3],
            b5=prediction_result.borrow_vector[4],
            b6=prediction_result.borrow_vector[5],
            compile_success=prediction_result.actual_compile,
            template_name=template_name,
            source=prediction_result.source,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.record(sample)

    def load_feedback(self) -> List[FeedbackSample]:
        """Load all feedback samples from the JSONL file."""
        if not os.path.exists(self._path):
            return []
        samples = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                samples.append(FeedbackSample(**{
                    k: d[k] for k in FeedbackSample.__dataclass_fields__
                    if k in d
                }))
        return samples

    def count(self) -> int:
        """Number of feedback samples stored."""
        return len(self.load_feedback())

    def retrain(
        self,
        metrics_path: str = METRICS_JSONL,
        min_auc: float = 0.85,
    ) -> Tuple[Optional[LogisticRegression], Optional[StandardScaler], float]:
        """Retrain classifier merging original metrics + feedback.

        Returns (clf, scaler, loo_cv_auc). Returns (None, None, 0.0) if
        LOO-CV AUC < min_auc threshold.
        """
        # Load original training data
        original = []
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        original.append(json.loads(line))

        # Load feedback
        feedback = self.load_feedback()

        # Merge into feature matrix
        all_samples = []
        for s in original:
            bv = [s["b1"], s["b2"], s["b3"], s["b4"], s["b5"], s.get("b6", 0.0)]
            eb = float(np.dot(WEIGHTS, bv))
            all_samples.append((bv + [eb], int(s["compile_success"])))

        for s in feedback:
            bv = [s.b1, s.b2, s.b3, s.b4, s.b5, s.b6]
            eb = float(np.dot(WEIGHTS, bv))
            all_samples.append((bv + [eb], int(s.compile_success)))

        if len(all_samples) < 10:
            return None, None, 0.0

        X = np.array([s[0] for s in all_samples])
        y = np.array([s[1] for s in all_samples])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)

        # LOO-CV AUC
        cv = min(len(all_samples), 10)  # k-fold (up to LOO)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
        auc = float(scores.mean())

        if auc < min_auc:
            return None, None, auc

        clf.fit(X_scaled, y)
        return clf, scaler, auc
