"""ResonanceAgent: reads golden_resonance_matrix, fires when any pair < 0.75.

Identifies which structural change would bring pair toward 51.8°.
Dispatches to structural_agent. Model: qwen3:8b.
"""
import numpy as np
from tensor.agent_network import AgentNode, AgentProposal, PHI

GOLDEN_ANGLE_COS = 1.0 / PHI


class ResonanceAgent(AgentNode):
    def __init__(self):
        super().__init__(
            role='resonance',
            model='qwen3:8b',
            level='code',
            poll_interval=15.0,
        )

    def should_fire(self, context: dict) -> bool:
        grm = context.get('golden_resonance_matrix', [])
        if not grm:
            return False
        for row in grm:
            for val in row:
                if isinstance(val, (int, float)) and val < 0.75:
                    return True
        return False

    def generate_change(self, context: dict) -> AgentProposal:
        grm = context.get('golden_resonance_matrix', [])
        # Find weakest pair
        min_val, min_pair = 1.0, (0, 0)
        for i, row in enumerate(grm):
            for j, val in enumerate(row):
                if i != j and isinstance(val, (int, float)) and val < min_val:
                    min_val = val
                    min_pair = (i, j)

        return AgentProposal(
            agent_role=self.role,
            target_level=self.level,
            description=(f"Improve resonance L{min_pair[0]}↔L{min_pair[1]} "
                         f"from {min_val:.4f} toward golden angle"),
            predicted_delta=0.005,
        )
