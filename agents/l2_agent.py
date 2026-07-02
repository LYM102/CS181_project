"""L2 agent: belief-augmented SARSA with residual action gating."""

from agents.belief_sarsa_agent import BeliefSarsaAgent


class L2Agent(BeliefSarsaAgent):
    def __init__(self, name: str = "L2", **kwargs):
        kwargs.setdefault("use_action_gating", True)
        kwargs.setdefault("gate_selective", False)
        kwargs.setdefault("gate_scale", 1.0)
        super().__init__(name=name, **kwargs)
