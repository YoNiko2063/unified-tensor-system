"""
FICUTS Self-Expanding Network — GrowingNeuralNetwork

Wraps UnifiedTensorNetwork so it can use its own predictions to decide
when to add new mode heads (self-directed expansion).

Growth algorithm:
  1. After each forward pass, compute per-head reconstruction error
     (how well each mode head predicts from its subspace).
  2. If max head error exceeds threshold AND dormant heads exist:
     a. Find the overloaded head (highest error).
     b. Promote the next dormant domain by initialising its mode head
        with weights copied from the overloaded head + small perturbation.
     c. Record growth event in history.
  3. Repeat: the network grows organically as new domains are encountered.

Expansion principle compliance:
  - Preallocate: all 150 mode heads exist from the start (dormant = zero bias)
  - Reserve dummy heads: dormant heads have near-zero domain_bias until promoted
  - Dynamic basis: IntegratedHDVSystem registers new domain mask on promotion
  - Curvature-triggered: DynamicExpander calls self_directed_grow on ρ-spike
  - Spectral hashing: DomainRegistry hashes frequencies to unused HDV indices
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class GrowingNeuralNetwork:
    """
    Wrapper around UnifiedTensorNetwork that enables self-directed growth.

    The network uses its own per-head reconstruction errors to decide when
    capacity in a specific mode is exhausted, then promotes the most similar
    dormant mode head to help, initialised via weight transfer (not random).
    """

    def __init__(
        self,
        network,                    # UnifiedTensorNetwork instance
        domain_registry,            # DomainRegistry (150 domains)
        hdv_system=None,            # IntegratedHDVSystem (for domain registration)
        growth_error_threshold: float = 0.5,
        max_growth_per_session: int = 20,
    ):
        self.net = network
        self.registry = domain_registry
        self.hdv_system = hdv_system
        self.growth_threshold = growth_error_threshold
        self.max_growth = max_growth_per_session

        self._growth_history: List[Dict] = []
        self._head_error_ema: Dict[int, float] = {}   # exponential moving avg
        self._ema_alpha = 0.1

        # Mark which heads are "promoted" (have been grown into)
        self._promoted_heads: set = set(range(
            min(13, self.net.n_modes)
        ))  # first 13 are pre-active

    # ── Error monitoring ───────────────────────────────────────────────────────

    def compute_head_errors(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Dict[int, float]:
        """
        Estimate per-head reconstruction error.

        If targets provided: use MSE(head_output, target).
        Else: use MSE(head_output, mean_output) as proxy for head disagreement.

        x:       [B, embed_dim] input tensor
        targets: [B, embed_dim] optional target

        Returns dict: head_index → scalar error.
        """
        self.net.eval()
        with torch.no_grad():
            try:
                # Run each mode head independently
                B = x.shape[0]
                errors: Dict[int, float] = {}
                mean_output = None

                head_outputs = []
                for i, head in enumerate(self.net.mode_heads):
                    out = head(x)          # [B, embed_dim]
                    head_outputs.append(out)

                # Reference: mean across all heads
                stack = torch.stack(head_outputs, dim=0)  # [n_modes, B, E]
                mean_output = stack.mean(dim=0)           # [B, E]

                for i, out in enumerate(head_outputs):
                    if targets is not None:
                        err = float(nn.functional.mse_loss(out, targets).item())
                    else:
                        err = float(nn.functional.mse_loss(out, mean_output).item())
                    errors[i] = err

                return errors
            except Exception:
                return {}

    def update_ema_errors(self, errors: Dict[int, float]):
        """Update exponential moving averages of per-head errors."""
        for i, err in errors.items():
            prev = self._head_error_ema.get(i, err)
            self._head_error_ema[i] = self._ema_alpha * err + (1 - self._ema_alpha) * prev

    def worst_head(self) -> Tuple[int, float]:
        """Return (head_index, error) for the most overloaded head."""
        if not self._head_error_ema:
            return 0, 0.0
        worst_idx = max(self._head_error_ema, key=self._head_error_ema.__getitem__)
        return worst_idx, self._head_error_ema[worst_idx]

    # ── Growth decision ────────────────────────────────────────────────────────

    def should_grow(self) -> bool:
        """
        Return True if:
          - Max EMA error > growth_threshold (some head is overloaded)
          - Dormant domain heads exist (room to grow)
          - Session growth budget not exhausted
        """
        if len(self._growth_history) >= self.max_growth:
            return False
        inactive = self.registry.inactive_domains()
        if not inactive:
            return False
        _, worst_err = self.worst_head()
        return worst_err > self.growth_threshold

    # ── Growth execution ───────────────────────────────────────────────────────

    def grow(self, from_head_idx: Optional[int] = None, domain_hint: str = "") -> Optional[int]:
        """
        Promote one dormant domain head:
          1. Select the most relevant inactive domain (keyword match if hint given)
          2. Activate it in DomainRegistry + HDV system
          3. Initialise new head weights from the overloaded head + small noise

        Returns new head index, or None if growth is not possible.
        """
        inactive = self.registry.inactive_domains()
        if not inactive:
            return None

        # Pick new domain: best keyword match if hint given, else next in list
        if domain_hint:
            matches = self.registry.classify_domain(domain_hint, top_k=3)
            new_domain = next(
                (m for m in matches if m in inactive), inactive[0]
            )
        else:
            new_domain = inactive[0]

        # Activate domain in registry (allocates HDV mask)
        new_head_idx = self.registry.activate_domain(new_domain, self.hdv_system)
        if new_head_idx < 0 or new_head_idx >= self.net.n_modes:
            return None

        # Determine source head for weight transfer
        if from_head_idx is None or from_head_idx >= self.net.n_modes:
            from_head_idx, _ = self.worst_head()

        # Weight transfer: copy from overloaded head + perturbation
        self._transfer_weights(from_head_idx, new_head_idx)
        self._promoted_heads.add(new_head_idx)

        event = {
            "step": len(self._growth_history),
            "timestamp": time.time(),
            "from_head": from_head_idx,
            "new_head": new_head_idx,
            "new_domain": new_domain,
            "new_domain_display": self.registry.get_display_name(new_domain),
            "worst_error_at_growth": self._head_error_ema.get(from_head_idx, 0.0),
        }
        self._growth_history.append(event)

        print(f"[Growth] Promoted domain '{new_domain}' → head {new_head_idx} "
              f"(from head {from_head_idx}, err={event['worst_error_at_growth']:.3f})")
        return new_head_idx

    def _transfer_weights(self, from_idx: int, to_idx: int, noise_scale: float = 0.01):
        """Copy weights from mode_heads[from_idx] to mode_heads[to_idx] + small noise."""
        src = self.net.mode_heads[from_idx]
        dst = self.net.mode_heads[to_idx]
        with torch.no_grad():
            for (_, src_p), (_, dst_p) in zip(
                src.named_parameters(), dst.named_parameters()
            ):
                noise = torch.randn_like(src_p.data) * noise_scale
                dst_p.data.copy_(src_p.data + noise)

    # ── Self-directed grow ─────────────────────────────────────────────────────

    def self_directed_grow(
        self,
        x: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        domain_hint: str = "",
    ) -> Optional[int]:
        """
        Full autonomous growth cycle:
          1. Compute per-head errors (if x given)
          2. Update EMAs
          3. Check should_grow()
          4. If yes: grow() and return new head idx; else return None.
        """
        if x is not None:
            errors = self.compute_head_errors(x, targets)
            if errors:
                self.update_ema_errors(errors)

        if not self.should_grow():
            return None

        from_idx, _ = self.worst_head()
        return self.grow(from_head_idx=from_idx, domain_hint=domain_hint)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def growth_history(self) -> List[Dict]:
        return list(self._growth_history)

    def status(self) -> Dict:
        worst_idx, worst_err = self.worst_head()
        return {
            "total_heads": self.net.n_modes,
            "promoted_heads": len(self._promoted_heads),
            "growth_events": len(self._growth_history),
            "worst_head": worst_idx,
            "worst_head_error": round(worst_err, 4),
            "should_grow": self.should_grow(),
            "active_domains": self.registry.n_active(),
            "inactive_domains": self.registry.n_inactive(),
        }


# ── UnifiedTensorNetwork extension ────────────────────────────────────────────

def add_mode_head_to_network(network, domain: str = "new_domain") -> int:
    """
    Append a new ModeHead to an existing UnifiedTensorNetwork.

    Used when n_modes needs to grow beyond the initial allocation.
    Returns the new head's index.

    Note: for the 150-domain case, all mode heads already exist — this
    function is for cases where n_modes itself must grow beyond 150.
    """
    from tensor.unified_network import ModeHead

    new_idx = len(network.mode_heads)
    new_head = ModeHead(
        mode_id=new_idx,
        domain=domain,
        embed_dim=network.mode_heads[0].transform[0].out_features,
    ).to(next(network.parameters()).device)

    # nn.ModuleList.append registers the new module correctly
    network.mode_heads.append(new_head)
    network.n_modes = len(network.mode_heads)

    return new_idx
