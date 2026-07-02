# agents/aggressive_agent.py - Exploitative bluff/trap opponent for BNN + gating eval
"""
AggressiveAgent wraps ExpertAgent (CFR) with deliberate deviations:

  - Weak hand (φ < HAND_STRENGTH_WEAK): bluff-raise with elevated probability
  - Strong hand (φ > HAND_STRENGTH_STRONG): value-raise or slow-play (trap call)
  - Medium: CFR base policy

Thresholds align with treys hand-strength labels (evaluator.HAND_STRENGTH_*).
See paper Appendix (app:aggressive) for full protocol.
"""

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
        strength = obs.equity  # treys hand strength in [0, 1]

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


class TightPassiveAgent(BaseAgent):
    """Tight-passive contrast opponent (BNN training diversity)."""

    def __init__(self, name: str = "TightPassiveAgent"):
        super().__init__(name=name)

    def act(self, obs: Observation) -> int:
        legal_actions = obs.legal_actions
        strength = obs.equity

        if strength > HAND_STRENGTH_STRONG + 0.08:
            return RAISE if RAISE in legal_actions else CALL
        if strength > HAND_STRENGTH_WEAK:
            return CALL if CALL in legal_actions else FOLD
        if strength > HAND_STRENGTH_WEAK - 0.12:
            if obs.current_bet <= 10:
                return CALL if CALL in legal_actions else FOLD
            return FOLD if FOLD in legal_actions else CALL
        return FOLD if FOLD in legal_actions else CALL

    def update(self, obs, action, reward, next_obs, done):
        pass
