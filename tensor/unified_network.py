"""
FICUTS Layer 9 (v2.1): Unified Continuous Network — 150 internal modes

Classes:
  - ModeHead              : domain-specialist sub-network within unified net  (Task 9.1)
  - UnifiedTensorNetwork  : single network, modes coupled via cross-attention  (Task 9.1)
  - compute_isometric_loss: differentiable distance-preservation loss          (Task 9.2)
  - compute_lyapunov_energy: global energy over unified state space            (Task 9.2)
  - UnifiedNetworkTrainer : training loop with passive learning                (Task 9.3)

Key principle: ONE network, NOT 150 separate models.
Gradients flow through the entire network continuously — ECE mode learns from
biology data via shared parameters and cross-attention (passive learning).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# 13 named domains + generics to reach 150 total
DOMAINS = [
    'ece', 'biology', 'finance', 'physics', 'chemistry',
    'materials', 'aerospace', 'civil_eng', 'mechanical',
    'quantum', 'neuroscience', 'genomics', 'drug_discovery',
] + [f'domain_{i}' for i in range(137)]


# ── Task 9.1: Architecture ────────────────────────────────────────────────────

class ModeHead(nn.Module):
    """
    Domain-specialist sub-network within the unified network.
    NOT a separate model — shares gradients through parent.
    """

    def __init__(self, mode_id: int, domain: str, embed_dim: int = 512):
        super().__init__()
        self.mode_id = mode_id
        self.domain = domain

        self.transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        self.domain_bias = nn.Parameter(torch.randn(embed_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x) + self.domain_bias


class UnifiedTensorNetwork(nn.Module):
    """
    Single neural network with n_modes internal modes.

    Architecture:
      shared HDV embedding → 150 ModeHeads → cross-mode attention
      → mode gating → LayerNorm → universal HDV decoder

    Passive learning: gradients flow to ALL modes on every backward pass,
    even modes not active for the current input domain.
    """

    def __init__(self, hdv_dim: int = 10000, n_modes: int = 150,
                 embed_dim: int = 512):
        super().__init__()
        self.hdv_dim = hdv_dim
        self.n_modes = n_modes
        self.embed_dim = embed_dim

        self.hdv_embedding = nn.Embedding(hdv_dim, embed_dim)

        self.mode_heads = nn.ModuleList([
            ModeHead(mode_id=i, domain=DOMAINS[i % len(DOMAINS)],
                     embed_dim=embed_dim)
            for i in range(n_modes)
        ])

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, hdv_dim),
        )

        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor,
                active_modes: Optional[List[int]] = None) -> torch.Tensor:
        """
        Forward pass.

        x            : [batch, seq_len] — HDV indices
        active_modes : which modes are active (default: all)
        returns      : [batch, hdv_dim]

        ALL modes run; passive learning occurs via shared gradients.
        """
        if active_modes is None:
            active_modes = list(range(self.n_modes))

        # Embed → dense
        embedded = self.hdv_embedding(x)      # [B, seq, E]
        embedded = embedded.mean(dim=1)        # [B, E]

        # All mode heads (gradients flow to every head)
        mode_outputs = torch.stack(
            [head(embedded) for head in self.mode_heads], dim=1
        )  # [B, n_modes, E]

        # Cross-mode attention (passive learning mechanism)
        attended, _ = self.cross_attention(
            mode_outputs, mode_outputs, mode_outputs
        )  # [B, n_modes, E]

        # Gate: only active modes contribute to output, but gradients
        # still flow through attended (which sees all modes)
        mask = torch.zeros(self.n_modes, device=self.device)
        mask[active_modes] = 1.0
        gated = attended * mask.unsqueeze(0).unsqueeze(2)  # [B, n_modes, E]

        aggregated = self.norm(gated.sum(dim=1))           # [B, E]
        return self.decoder(aggregated)                    # [B, hdv_dim]


# ── Task 9.2: Isometric + Lyapunov ───────────────────────────────────────────

def compute_isometric_loss(network: UnifiedTensorNetwork,
                           data_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                           active_modes: List[int]) -> torch.Tensor:
    """
    Isometric regularization: ||x1-x2||_input ≈ ||f(x1)-f(x2)||_hdv

    Continuous and differentiable — gradients reach all modes.
    """
    total = torch.tensor(0.0, requires_grad=True)
    for x1, x2 in data_pairs:
        hdv1 = network(x1.unsqueeze(0), active_modes)
        hdv2 = network(x2.unsqueeze(0), active_modes)
        input_dist = torch.norm(x1.float() - x2.float())
        hdv_dist = torch.norm(hdv1 - hdv2)
        total = total + (input_dist - hdv_dist) ** 2
    return total / max(len(data_pairs), 1)


def compute_lyapunov_energy(network: UnifiedTensorNetwork,
                            state: torch.Tensor) -> torch.Tensor:
    """
    Global Lyapunov energy E(θ) = α·||θ||² + β·coupling_energy.

    Only possible with a unified network (continuous state space).
    Enables hardware co-design from network dynamics.
    """
    param_energy = sum(torch.sum(p ** 2) for p in network.parameters())

    mode_outputs = torch.stack(
        [head(state) for head in network.mode_heads]
    )  # [n_modes, embed_dim]

    coupling_matrix = torch.corrcoef(mode_outputs)
    coupling_energy = torch.sum(torch.abs(coupling_matrix))

    alpha, beta = 0.001, 0.01
    return alpha * param_energy + beta * coupling_energy


# ── Task 9.3: Training Loop ───────────────────────────────────────────────────

class UnifiedNetworkTrainer:
    """
    Train unified network with passive learning.

    When training on ECE data, biology mode ALSO learns via shared
    gradients and cross-attention (not just active-mode gradients).
    """

    def __init__(self, network: UnifiedTensorNetwork, learning_rate: float = 1e-4):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train_step(self, batch: torch.Tensor, active_domain: str,
                   target: torch.Tensor) -> float:
        active_modes = [i for i, d in enumerate(DOMAINS) if d == active_domain]
        if not active_modes:
            active_modes = [0]

        self.optimizer.zero_grad()
        output = self.network(batch, active_modes)
        loss = torch.mean((output - target) ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def verify_passive_learning(self, active_domain: str,
                                inactive_domain: str) -> bool:
        """Return True if inactive mode's params changed after one train step."""
        inactive_id = DOMAINS.index(inactive_domain)

        params_before = [p.clone().detach()
                         for p in self.network.mode_heads[inactive_id].parameters()]

        batch = torch.randint(0, self.network.hdv_dim, (4, 10))
        target = torch.randn(4, self.network.hdv_dim)
        self.train_step(batch, active_domain, target)

        for p_before, p_after in zip(
            params_before,
            self.network.mode_heads[inactive_id].parameters()
        ):
            if not torch.equal(p_before, p_after):
                return True
        return False
