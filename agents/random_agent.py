# agents/random_agent.py - Random strategy Agent (for testing and baseline)

import random
from agents.base_agent import BaseAgent
from game.engine import Observation
from game.constants import CALL, RAISE


class RandomAgent(BaseAgent):
    """Agent that randomly selects legal actions, for testing and baseline comparison"""

    def __init__(self, name: str = "RandomAgent", fold_prob: float = 0.1):
        """
        Args:
            name: Agent name
            fold_prob: fold probability (only when FOLD is a legal action)
        """
        super().__init__(name=name)
        self.fold_prob = fold_prob

    def act(self, obs: Observation) -> int:
        legal = obs.legal_actions
        # Simple strategy: fold with some probability, otherwise random among CALL/RAISE
        if CALL in legal and RAISE in legal:
            if random.random() < self.fold_prob and 0 in legal:
                return 0  # FOLD
            return random.choice([CALL, RAISE])
        # Can only CALL (raise limit reached)
        return CALL if CALL in legal else legal[0]
