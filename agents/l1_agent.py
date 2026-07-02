"""L1 agent: belief-augmented SARSA (state-level belief injection)."""

from agents.belief_sarsa_agent import BeliefSarsaAgent


class L1Agent(BeliefSarsaAgent):
    def __init__(self, name: str = "L1", **kwargs):
        kwargs.setdefault("use_action_gating", False)
        super().__init__(name=name, **kwargs)
