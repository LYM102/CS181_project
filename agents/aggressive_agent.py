# agents/aggressive_agent.py - Aggressive bluffing agent for BNN training diversity
"""
AggressiveAgent wraps ExpertAgent with increased bluffing behavior.

Purpose:
  - Provide bluff-rich training data for BNN opponent modeling.
  - When the agent has a weak hand, it raises with elevated probability (bluff).
  - When the agent has a strong hand, it also raises more (over-value-betting).
  - This creates data where the BNN must distinguish bluffs from value bets
    using behavioral patterns (action sequences, board texture, bet sizing).

Key parameters:
  - bluff_raise_prob: probability of raising with weak hand (equity < 0.3)
  - value_raise_prob: probability of raising with strong hand (equity > 0.6)
  - slowplay_prob: probability of just calling with strong hand (trap)
"""

from __future__ import annotations
import random

from agents.base_agent import BaseAgent
from agents.expert_agent import ExpertAgent
from game.engine import Observation
from game.constants import FOLD, CALL, RAISE


class AggressiveAgent(BaseAgent):
    """
    Aggressive bluffing agent that over-raises relative to Nash equilibrium.

    Combines ExpertAgent's CFR-based strategy with additional bluff/value
    raise overrides based on hand equity thresholds.

    Bluffing behavior:
      - Weak hands (equity < 0.3): raises with bluff_raise_prob (default 0.45)
      - Medium hands (0.3-0.6): plays ExpertAgent strategy
      - Strong hands (equity > 0.6): raises with value_raise_prob (default 0.8)
      - Occasionally slowplays strong hands (just calls) to mix strategy
    """

    def __init__(self, name: str = "AggressiveAgent",
                 bluff_raise_prob: float = 0.45,
                 value_raise_prob: float = 0.80,
                 slowplay_prob: float = 0.15,
                 policy_path: str = None):
        super().__init__(name=name)
        self.bluff_raise_prob = bluff_raise_prob
        self.value_raise_prob = value_raise_prob
        self.slowplay_prob = slowplay_prob

        # Use ExpertAgent's CFR policy as base
        self._expert = ExpertAgent(name=f"{name}_base", policy_path=policy_path)

    def act(self, obs: Observation) -> int:
        """
        Action selection with overridden bluffing/value-betting frequencies.
        """
        legal_actions = obs.legal_actions
        equity = obs.equity

        # --- Weak hand: bluff with elevated probability ---
        if equity < 0.3:
            if RAISE in legal_actions and random.random() < self.bluff_raise_prob:
                return RAISE
            # If not bluffing, still use expert strategy (may fold or call)
            return self._expert.act(obs)

        # --- Strong hand: over-value-bet or slowplay ---
        elif equity > 0.6:
            # Slowplay: just call with strong hand (trap)
            if random.random() < self.slowplay_prob:
                return CALL if CALL in legal_actions else RAISE
            # Value raise
            if RAISE in legal_actions and random.random() < self.value_raise_prob:
                return RAISE
            return CALL if CALL in legal_actions else self._expert.act(obs)

        # --- Medium hand: defer to ExpertAgent (Nash equilibrium) ---
        else:
            return self._expert.act(obs)

    def update(self, obs, action, reward, next_obs, done):
        pass


class TightPassiveAgent(BaseAgent):
    """
    Tight-passive agent (counter-style to aggressive).

    Only raises with very strong hands, folds marginal situations.
    Provides contrast data for BNN training.
    """

    def __init__(self, name: str = "TightPassiveAgent"):
        super().__init__(name=name)

    def act(self, obs: Observation) -> int:
        legal_actions = obs.legal_actions
        equity = obs.equity

        if equity > 0.7:
            # Very strong: raise
            if RAISE in legal_actions:
                return RAISE
            return CALL
        elif equity > 0.45:
            # Decent hand: call
            return CALL if CALL in legal_actions else FOLD
        elif equity > 0.3:
            # Marginal: call only if cheap
            if obs.current_bet <= 10:
                return CALL if CALL in legal_actions else FOLD
            return FOLD if FOLD in legal_actions else CALL
        else:
            # Weak: fold
            return FOLD if FOLD in legal_actions else CALL

    def update(self, obs, action, reward, next_obs, done):
        pass
