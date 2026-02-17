"""AgentNetwork: field-reading agent population replacing GSD hierarchy.

Each agent reads /tmp/tensor_context, fires when its free energy condition
is met, acts, and the tensor validates. No coordinator. No job queue.
The field is the governance.

Includes PredictiveLayer (L3) and MetaLoss integration (L5).

FICUTS Layer 4:
  - Task 4.1: RLock (_state_lock) protecting agent reads/writes in run_cycle
"""
import json
import subprocess
import sys
import threading
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PHI = 1.6180339887


@dataclass
class AgentProposal:
    agent_role: str
    target_level: str
    description: str
    predicted_delta: float = 0.0
    predicted_meta_delta: float = 0.0
    diff: str = ''


@dataclass
class AgentNode:
    role: str
    model: str
    level: str
    influence: float = 1.0
    poll_interval: float = 10.0
    correct_predictions: int = 0
    total_predictions: int = 0

    def firing_threshold(self, context: dict) -> float:
        """Tension threshold for firing: 1/PHI ≈ 0.618 of max tension."""
        return 1.0 / PHI  # ~0.618 — fire when tension > 38.2% of scale

    def should_fire(self, context: dict) -> bool:
        consonance = context.get('consonance', {})
        level_cons = consonance.get(self.level, 1.0)
        tension = 1.0 - level_cons
        return tension > (1.0 - self.firing_threshold(context))

    def generate_change(self, context: dict) -> AgentProposal:
        """Generate a change proposal. Override in subclasses."""
        return AgentProposal(
            agent_role=self.role,
            target_level=self.level,
            description=f"{self.role} action on {self.level}",
        )

    def predict_delta(self, proposal: AgentProposal) -> float:
        """Predict consonance delta if proposal is applied."""
        return proposal.predicted_delta

    def predict_meta_delta(self, proposal: AgentProposal,
                           trajectory) -> float:
        """Predict change in d²/dt² if proposal is accepted."""
        if trajectory is None:
            return proposal.predicted_delta
        current_accel = trajectory.consonance_acceleration(self.level)
        # Predict the proposal will maintain current acceleration + delta
        return proposal.predicted_meta_delta if proposal.predicted_meta_delta != 0 else (
            proposal.predicted_delta * (1.0 + abs(current_accel)))

    def update_influence(self, predicted: float, actual: float):
        """Hebbian update: accurate predictions increase influence."""
        self.total_predictions += 1
        error = abs(predicted - actual)
        if error < 0.01:
            self.correct_predictions += 1
            self.influence = min(self.influence * PHI, 10.0)
        else:
            self.influence *= (1.0 / PHI)
            self.influence = max(self.influence, 0.01)


class PredictiveLayer:
    """Agents that predict tensor state N steps after their action.

    Prediction error is the learning signal that tells the system where
    it is most ignorant. High ignorance + accelerating consonance = where to focus.
    """

    def __init__(self, trajectory=None):
        self.trajectory = trajectory
        self.error_history: Dict[str, List[float]] = defaultdict(list)

    def record_prediction(self, level: str,
                          predicted: float, actual: float):
        error = abs(predicted - actual)
        self.error_history[level].append(error)
        # Keep bounded
        if len(self.error_history[level]) > 500:
            self.error_history[level] = self.error_history[level][-500:]

    def ignorance_map(self) -> Dict[str, float]:
        """Mean prediction error per level. High = system does not understand this level."""
        return {l: float(np.mean(e[-50:])) if e else 0.0
                for l, e in self.error_history.items()}

    def learning_priority(self) -> List[str]:
        """argmax(ignorance * consonance_acceleration).

        Where the system is most ignorant AND improving fastest.
        This is the curriculum — what to learn next.
        """
        ig = self.ignorance_map()
        if not ig or self.trajectory is None:
            return sorted(ig.keys())
        acc = {l: self.trajectory.consonance_acceleration(l) for l in ig}
        score = {l: ig[l] * max(acc[l], 0) for l in ig}
        return sorted(score, key=lambda l: score[l], reverse=True)


class AgentNetwork:
    """Population of field-reading agents. No coordinator — the field is the governance."""

    def __init__(self, tensor=None, trajectory=None):
        self.tensor = tensor
        self.trajectory = trajectory
        self.agents: List[AgentNode] = []
        self.predictive = PredictiveLayer(trajectory)
        self._apply_fn: Optional[Callable] = None
        self._revert_fn: Optional[Callable] = None
        self._cycle_count = 0
        self._state_lock = threading.RLock()   # Task 4.1: protect agent state

    def add_agent(self, agent: AgentNode):
        self.agents.append(agent)

    def set_apply_fn(self, fn: Callable):
        """Set function to apply a proposal: fn(proposal) -> bool."""
        self._apply_fn = fn

    def set_revert_fn(self, fn: Callable):
        """Set function to revert a proposal: fn(proposal) -> None."""
        self._revert_fn = fn

    def _read_context(self) -> dict:
        """Read /tmp/tensor_context."""
        try:
            with open('/tmp/tensor_context') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def run_cycle(self) -> Optional[dict]:
        """One agent network cycle: read context, fire, arbitrate, apply, validate.

        Task 4.1: atomic read of agent state, no lock during slow operations,
        atomic write of agent state on completion.
        """
        with self._state_lock:
            self._cycle_count += 1
            cycle = self._cycle_count

        context = self._read_context()
        if not context:
            return None

        # Record trajectory (trajectory has its own write lock)
        if self.trajectory is not None:
            self.trajectory.record(context)

        # Atomic read: snapshot which agents should fire
        with self._state_lock:
            firing = [a for a in self.agents if a.should_fire(context)]
        if not firing:
            return {'status': 'idle', 'cycle': cycle}

        # No lock during proposal generation (can be slow/LLM calls)
        proposals = []
        for a in firing:
            p = a.generate_change(context)
            p.predicted_delta = a.predict_delta(p)
            proposals.append((a, p))

        # Arbitrate: highest predicted_meta_delta * influence wins
        with self._state_lock:
            winner_agent, winner_proposal = max(
                proposals,
                key=lambda ap: ap[0].predict_meta_delta(
                    ap[1], self.trajectory) * ap[0].influence
            )
            cons_before = context.get('consonance', {}).get(
                winner_agent.level, 0.0)

        # Apply change (outside lock — external operation)
        if self._apply_fn is not None:
            self._apply_fn(winner_proposal)

        # Read new context
        context_after = self._read_context()
        cons_after = context_after.get('consonance', {}).get(
            winner_agent.level, cons_before)
        delta = cons_after - cons_before

        # Atomic write: update agent state
        with self._state_lock:
            winner_agent.update_influence(winner_proposal.predicted_delta, delta)
            self.predictive.record_prediction(
                winner_agent.level,
                winner_proposal.predicted_delta,
                delta)

            if delta <= 0 and self._revert_fn is not None:
                self._revert_fn(winner_proposal)

            if cycle % 10 == 0:
                priorities = self.predictive.learning_priority()
                for agent in self.agents:
                    if agent.level in priorities[:2]:
                        agent.poll_interval = max(2.0, agent.poll_interval * 0.8)
                    else:
                        agent.poll_interval = min(30.0, agent.poll_interval * 1.1)

        return {
            'status': 'applied' if delta > 0 else 'reverted',
            'cycle': cycle,
            'agent': winner_agent.role,
            'predicted_delta': winner_proposal.predicted_delta,
            'actual_delta': delta,
            'influence': winner_agent.influence,
        }
