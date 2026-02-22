"""Template registry — stores and retrieves Rust code generation templates.

Each template is a callable that takes a parameter dict and returns Rust source.
Templates are ranked by E_borrow (ascending) for safety-first selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from codegen.intent_spec import BorrowProfile, IntentSpec, e_borrow


@dataclass
class RustTemplate:
    """A registered Rust code generation template.

    Attributes:
        name: Unique template identifier.
        domain: Problem domain this template serves.
        operation: Specific operation implemented.
        borrow_profile: Expected ownership pattern.
        design_bv: Design-time BorrowVector (B1..B6).
        render: Callable taking parameters dict, returning Rust source string.
        requires_cargo: Whether the template needs external crates (reqwest, serde).
        crate_deps: Cargo.toml [dependencies] entries.
        description: Human-readable description.
    """
    name: str
    domain: str
    operation: str
    borrow_profile: BorrowProfile
    design_bv: Tuple[float, ...]
    render: Callable[[Dict], str]
    requires_cargo: bool = False
    crate_deps: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def e_borrow(self) -> float:
        return e_borrow(self.design_bv)


class TemplateRegistry:
    """Expandable catalog of Rust code generation templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, RustTemplate] = {}

    def register(self, template: RustTemplate) -> None:
        """Register a template by name."""
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[RustTemplate]:
        """Look up template by exact name."""
        return self._templates.get(name)

    def lookup(self, domain: str, operation: str) -> Optional[RustTemplate]:
        """Find template matching domain + operation."""
        for t in self._templates.values():
            if t.domain == domain and t.operation == operation:
                return t
        return None

    def best_match(self, intent: IntentSpec) -> Optional[RustTemplate]:
        """Find best template for an IntentSpec, preferring lowest E_borrow.

        Matches by domain first, then operation. Among candidates, returns
        the one with lowest E_borrow (safest borrow pattern).
        """
        candidates = [
            t for t in self._templates.values()
            if t.domain == intent.domain and t.operation == intent.operation
        ]
        if not candidates:
            # Fall back to domain-only match
            candidates = [
                t for t in self._templates.values()
                if t.domain == intent.domain
            ]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.e_borrow)

    def all_templates(self) -> List[RustTemplate]:
        """Return all templates sorted by E_borrow ascending."""
        return sorted(self._templates.values(), key=lambda t: t.e_borrow)

    @property
    def count(self) -> int:
        return len(self._templates)


def register_all_templates(registry: "TemplateRegistry") -> None:
    """Register all built-in templates (numeric, market, html, physics) into a registry."""
    from codegen.templates import numeric_kernel, market_model, html_navigator, physics_sim
    numeric_kernel.register_all(registry)
    market_model.register_all(registry)
    html_navigator.register_all(registry)
    physics_sim.register_all(registry)
