"""Abstract base class for all agents."""

from abc import ABC, abstractmethod
from game.engine import Observation


class BaseAgent(ABC):


    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def act(self, obs: Observation) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def update(self, obs: Observation, action: int, reward: float,
               next_obs: Observation, done: bool) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
