"""ValidatorAgent: reactive classifier, accept/reject every change.

Always fires reactively after changes. Returns accept/reject with
eigenvalue evidence. Model: qwen2.5:1.5b (fast classifier).
"""
from tensor.agent_network import AgentNode, AgentProposal


class ValidatorAgent(AgentNode):
    def __init__(self):
        super().__init__(
            role='validator',
            model='qwen2.5:1.5b',
            level='code',
            poll_interval=5.0,
        )

    def should_fire(self, context: dict) -> bool:
        # Always fire reactively — validator checks every change
        return True

    def generate_change(self, context: dict) -> AgentProposal:
        consonance = context.get('consonance', {})
        gaps = context.get('eigenvalue_gaps', {})
        # Validator doesn't generate changes — it validates
        evidence = ', '.join(
            f"{k}={v}" for k, v in gaps.items())
        return AgentProposal(
            agent_role=self.role,
            target_level=self.level,
            description=f"Validate: eigenvalue_gaps=[{evidence}]",
            predicted_delta=0.0,
        )
