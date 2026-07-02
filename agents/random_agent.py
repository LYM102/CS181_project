"""Random baseline agent."""

import random
from agents.base_agent import BaseAgent
from game.engine import Observation
from game.constants import CALL, RAISE


class RandomAgent(BaseAgent):
    """Randomly selects legal actions with a small fold probability."""

    def __init__(self, name: str = "RandomAgent", fold_prob: float = 0.1):
        super().__init__(name=name)
        self.fold_prob = fold_prob

    def act(self, obs: Observation) -> int:
        legal = obs.legal_actions
        if CALL in legal and RAISE in legal:
            if random.random() < self.fold_prob and 0 in legal:
                return 0  # FOLD
            return random.choice([CALL, RAISE])
        return CALL if CALL in legal else legal[0]
