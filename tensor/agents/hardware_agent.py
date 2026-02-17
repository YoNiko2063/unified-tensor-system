"""HardwareAgent: reads L3 thermal/bandwidth coupling.

Fires when coupling > 0.85. Proposes scheduling changes to reduce
thermal coupling. Model: qwen2.5:1.5b.
"""
from tensor.agent_network import AgentNode, AgentProposal


class HardwareAgent(AgentNode):
    def __init__(self):
        super().__init__(
            role='hardware',
            model='qwen2.5:1.5b',
            level='hardware',
            poll_interval=30.0,
        )

    def should_fire(self, context: dict) -> bool:
        consonance = context.get('consonance', {})
        hw_cons = consonance.get('hardware', 1.0)
        # Fire when hardware consonance indicates high coupling
        return hw_cons < 0.5

    def generate_change(self, context: dict) -> AgentProposal:
        return AgentProposal(
            agent_role=self.role,
            target_level=self.level,
            description="Propose scheduling changes to reduce thermal coupling",
            predicted_delta=0.001,
        )
