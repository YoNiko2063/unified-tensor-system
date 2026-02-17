"""StructuralAgent: reads L2, fires on free_energy threshold.

Primary code-improving agent. Uses deepseek-coder:6.7b.
Output: unified diff applied via run_autonomy_tick().
"""
from tensor.agent_network import AgentNode, AgentProposal, PHI


class StructuralAgent(AgentNode):
    def __init__(self):
        super().__init__(
            role='structural',
            model='deepseek-coder:6.7b',
            level='code',
            poll_interval=10.0,
        )

    def should_fire(self, context: dict) -> bool:
        consonance = context.get('consonance', {})
        code_cons = consonance.get('code', 1.0)
        # Fire when code consonance below threshold
        return code_cons < 0.75

    def generate_change(self, context: dict) -> AgentProposal:
        # Identify highest free-energy FIM priority module
        fim = context.get('fim_priorities', {})
        indices = fim.get('indices', [])
        phi_weights = fim.get('phi_weights', [])
        target = f"module_idx_{indices[0]}" if indices else "unknown"
        weight = phi_weights[0] if phi_weights else 0.0

        return AgentProposal(
            agent_role=self.role,
            target_level=self.level,
            description=f"Refactor {target} (phi_weight={weight:.4f})",
            predicted_delta=weight * 0.01,
        )
