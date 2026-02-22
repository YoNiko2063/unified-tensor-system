"""IntentProjector — HDV direction → IntentSpec mapping.

Projects HDV vectors onto reference template directions to find
the best-matching code generation intent. Minimum overlap threshold
of 0.3 prevents spurious matches.

Optionally wires CrossDimensionalDiscovery for "code_intent" dimension.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from codegen.intent_spec import BorrowProfile, IntentSpec


@dataclass
class ProjectionResult:
    """Result of projecting an HDV direction onto template space."""
    intent: Optional[IntentSpec]
    template_name: str
    similarity: float
    all_similarities: Dict[str, float]
    above_threshold: bool


class IntentProjector:
    """Map HDV directions to IntentSpecs via template reference vectors."""

    def __init__(
        self,
        encoder=None,
        discovery=None,
        similarity_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            encoder: SemanticFlowEncoder for encoding intents.
            discovery: Optional CrossDimensionalDiscovery for recording.
            similarity_threshold: Minimum overlap to accept a match.
        """
        self._encoder = encoder
        self._discovery = discovery
        self._threshold = similarity_threshold

        # Reference HDV vectors per template (name → vector)
        self._reference_vectors: Dict[str, np.ndarray] = {}
        # Template metadata (name → IntentSpec prototype)
        self._prototypes: Dict[str, IntentSpec] = {}

    def register_template(
        self,
        name: str,
        prototype: IntentSpec,
        reference_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Register a template with its HDV reference direction.

        If no reference_vector provided, encodes the prototype via encoder.
        """
        if reference_vector is not None:
            self._reference_vectors[name] = reference_vector / (np.linalg.norm(reference_vector) + 1e-15)
        elif self._encoder is not None:
            vec = self._encoder.encode_code_intent(prototype)
            self._reference_vectors[name] = vec / (np.linalg.norm(vec) + 1e-15)
        self._prototypes[name] = prototype

    def register_from_registry(self, registry) -> None:
        """Register all templates from a TemplateRegistry.

        Creates IntentSpec prototypes from template metadata.
        """
        for template in registry.all_templates():
            prototype = IntentSpec(
                domain=template.domain,
                operation=template.operation,
                estimated_borrow_profile=template.borrow_profile,
            )
            self.register_template(template.name, prototype)

    def project(
        self,
        hdv_direction: np.ndarray,
        context: Optional[Dict] = None,
    ) -> ProjectionResult:
        """Project HDV direction onto template space.

        Finds best-matching template by cosine similarity in HDV space.
        Returns ProjectionResult with the matched IntentSpec (or None
        if below threshold).
        """
        if len(self._reference_vectors) == 0:
            return ProjectionResult(
                intent=None,
                template_name="",
                similarity=0.0,
                all_similarities={},
                above_threshold=False,
            )

        # Normalize query
        query_norm = np.linalg.norm(hdv_direction)
        if query_norm < 1e-15:
            return ProjectionResult(
                intent=None,
                template_name="",
                similarity=0.0,
                all_similarities={},
                above_threshold=False,
            )
        query = hdv_direction / query_norm

        # Compute similarities
        similarities = {}
        for name, ref_vec in self._reference_vectors.items():
            # Cosine similarity (both already normalized)
            sim = float(np.dot(query, ref_vec[:len(query)]))
            similarities[name] = sim

        # Best match
        best_name = max(similarities, key=similarities.get)
        best_sim = similarities[best_name]
        above = best_sim >= self._threshold

        intent = None
        if above and best_name in self._prototypes:
            proto = self._prototypes[best_name]
            intent = IntentSpec(
                domain=proto.domain,
                operation=proto.operation,
                complexity_class=proto.complexity_class,
                required_traits=list(proto.required_traits),
                estimated_borrow_profile=proto.estimated_borrow_profile,
                hdv_direction=hdv_direction,
                confidence=best_sim,
                parameters=dict(context or {}),
            )

            # Record in CrossDimensionalDiscovery if available
            if self._discovery is not None:
                try:
                    self._discovery.record_pattern(
                        "code_intent",
                        hdv_direction,
                        {
                            "type": "intent_projection",
                            "template": best_name,
                            "similarity": best_sim,
                            "domain": proto.domain,
                            "operation": proto.operation,
                        },
                    )
                except Exception:
                    pass  # Discovery is optional

        return ProjectionResult(
            intent=intent,
            template_name=best_name if above else "",
            similarity=best_sim,
            all_similarities=similarities,
            above_threshold=above,
        )

    def project_batch(
        self,
        hdv_directions: np.ndarray,
        contexts: Optional[List[Dict]] = None,
    ) -> List[ProjectionResult]:
        """Project multiple HDV directions."""
        results = []
        for i, direction in enumerate(hdv_directions):
            ctx = contexts[i] if contexts and i < len(contexts) else None
            results.append(self.project(direction, ctx))
        return results

    @property
    def registered_templates(self) -> List[str]:
        return list(self._reference_vectors.keys())

    @property
    def template_count(self) -> int:
        return len(self._reference_vectors)
