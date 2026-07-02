"""Exploitative opponent with deliberate bluff/trap deviations from CFR."""

from __future__ import annotations
import random

from agents.base_agent import BaseAgent
from agents.expert_agent import ExpertAgent
from game.engine import Observation
from game.constants import FOLD, CALL, RAISE
from game.evaluator import HAND_STRENGTH_WEAK, HAND_STRENGTH_STRONG


class AggressiveAgent(BaseAgent):
    """CFR-based agent with exploitable bluff and slow-play frequencies."""

    def __init__(self, name: str = "AggressiveAgent",
                 bluff_raise_prob: float = 0.45,
                 value_raise_prob: float = 0.80,
                 slowplay_prob: float = 0.15,
                 policy_path: str = None):
        super().__init__(name=name)
        self.bluff_raise_prob = bluff_raise_prob
        self.value_raise_prob = value_raise_prob
        self.slowplay_prob = slowplay_prob
        self._expert = ExpertAgent(name=f"{name}_base", policy_path=policy_path)

    def act(self, obs: Observation) -> int:
        legal_actions = obs.legal_actions
        strength = obs.equity

        if strength < HAND_STRENGTH_WEAK:
            if RAISE in legal_actions and random.random() < self.bluff_raise_prob:
                return RAISE
            return self._expert.act(obs)

        if strength > HAND_STRENGTH_STRONG:
            if random.random() < self.slowplay_prob:
                return CALL if CALL in legal_actions else RAISE
            if RAISE in legal_actions and random.random() < self.value_raise_prob:
                return RAISE
            return CALL if CALL in legal_actions else self._expert.act(obs)

        return self._expert.act(obs)

    def update(self, obs, action, reward, next_obs, done):
        pass
