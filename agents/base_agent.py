# agents/base_agent.py - Agent abstract base class

from abc import ABC, abstractmethod
from game.engine import Observation


class BaseAgent(ABC):
    """Abstract base class for all Agents, defining a unified interface"""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def act(self, obs: Observation) -> int:
        """
        Select an action based on observation.

        Args:
            obs: current observation (Observation object)

        Returns:
            action: FOLD(0) / CALL(1) / RAISE(2)
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state at the start of a hand (optional override)"""
        pass

    def update(self, obs: Observation, action: int, reward: float,
               next_obs: Observation, done: bool) -> None:
        """
        Update policy after a hand ends (for online learning Agents).

        Args:
            obs: observation before action
            action: action taken
            reward: reward received
            next_obs: observation after action
            done: whether this hand is over
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
