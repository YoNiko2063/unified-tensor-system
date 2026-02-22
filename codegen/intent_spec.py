"""IntentSpec — semantic intent specification for Rust code generation.

Maps high-level domain operations to expected BorrowVector profiles,
enabling pre-gate validation before template rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# BorrowVector weights from code_gen_experiment.py
WEIGHTS = np.array([0.25, 0.18, 0.15, 0.17, 0.15, 0.10])


class BorrowProfile(Enum):
    """Canonical borrow profiles mapping intent → expected BV range."""
    PURE_FUNCTIONAL = "pure_functional"
    SHARED_REFERENCE = "shared_reference"
    MUTABLE_OUTPUT = "mutable_output"
    ASYNC_IO = "async_io"


# Profile → (B1, B2, B3, B4, B5, B6) expected values
_PROFILE_BV: Dict[BorrowProfile, Tuple[float, ...]] = {
    BorrowProfile.PURE_FUNCTIONAL:  (0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
    BorrowProfile.SHARED_REFERENCE: (0.20, 0.15, 0.00, 0.00, 0.00, 0.00),
    BorrowProfile.MUTABLE_OUTPUT:   (0.20, 0.00, 0.00, 0.30, 0.00, 0.00),
    BorrowProfile.ASYNC_IO:         (0.30, 0.10, 0.10, 0.00, 0.00, 0.00),
}


def expected_borrow_vector(profile: BorrowProfile) -> Tuple[float, ...]:
    """Return (B1..B6) for the given profile."""
    return _PROFILE_BV[profile]


def e_borrow(bv: Tuple[float, ...]) -> float:
    """Scalar borrow energy from 6-component BorrowVector."""
    return float(np.dot(WEIGHTS, bv))


@dataclass
class IntentSpec:
    """Semantic specification for a Rust code generation target.

    Attributes:
        domain: Problem domain (e.g. "numeric", "market", "text", "api").
        operation: Specific operation (e.g. "sma", "ema", "resonance_detector").
        complexity_class: Computational complexity category.
        required_traits: Rust traits the generated code needs.
        data_shapes: Input/output tensor shape descriptions.
        parameters: Domain-specific parameter dict passed to template.
        estimated_borrow_profile: Expected ownership pattern.
        hdv_direction: Optional HDV vector encoding this intent.
        confidence: Confidence in the intent extraction [0, 1].
    """
    domain: str
    operation: str
    complexity_class: str = "element_wise"
    required_traits: List[str] = field(default_factory=list)
    data_shapes: Dict[str, str] = field(default_factory=dict)
    parameters: Dict = field(default_factory=dict)
    estimated_borrow_profile: BorrowProfile = BorrowProfile.PURE_FUNCTIONAL
    hdv_direction: Optional[np.ndarray] = None
    confidence: float = 1.0

    def expected_borrow_vector(self) -> Tuple[float, ...]:
        """Return (B1..B6) for this intent's estimated profile."""
        return expected_borrow_vector(self.estimated_borrow_profile)

    def expected_e_borrow(self) -> float:
        """Scalar borrow energy for this intent's expected profile."""
        return e_borrow(self.expected_borrow_vector())

    def to_dict(self) -> Dict:
        """Serialize to dict (excluding hdv_direction numpy array)."""
        return {
            "domain": self.domain,
            "operation": self.operation,
            "complexity_class": self.complexity_class,
            "required_traits": self.required_traits,
            "data_shapes": self.data_shapes,
            "parameters": self.parameters,
            "estimated_borrow_profile": self.estimated_borrow_profile.value,
            "confidence": self.confidence,
        }
