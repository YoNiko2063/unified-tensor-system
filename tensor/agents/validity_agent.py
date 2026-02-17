"""ValidityAgent: reads L0 + validity scores.

Fires when sentiment_velocity high but consonance_delta low.
Reweights L0 update. Protects against misleading signals. Model: qwen3:8b.
"""
from tensor.agent_network import AgentNode, AgentProposal


class ValidityAgent(AgentNode):
    def __init__(self):
        super().__init__(
            role='validity',
            model='qwen3:8b',
            level='market',
            poll_interval=10.0,
        )

    def should_fire(self, context: dict) -> bool:
        stress = context.get('stress_nodes', [])
        for s in stress:
            if s.get('level') == 'market' and s.get('risk', 0) > 0.6:
                return True
        return False

    def generate_change(self, context: dict) -> AgentProposal:
        return AgentProposal(
            agent_role=self.role,
            target_level=self.level,
            description="Reweight L0 market signal to reduce misleading input",
            predicted_delta=0.002,
        )
