# agents/base_agent.py - Agent 抽象基类

from abc import ABC, abstractmethod
from game.engine import Observation


class BaseAgent(ABC):
    """所有 Agent 的抽象基类，定义统一接口"""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def act(self, obs: Observation) -> int:
        """
        根据观测选择动作。

        Args:
            obs: 当前观测 (Observation 对象)

        Returns:
            action: FOLD(0) / CALL(1) / RAISE(2)
        """
        raise NotImplementedError

    def reset(self) -> None:
        """在一手牌开始时重置内部状态 (可选覆写)"""
        pass

    def update(self, obs: Observation, action: int, reward: float,
               next_obs: Observation, done: bool) -> None:
        """
        在一手牌结束后更新策略 (用于在线学习 Agent)。

        Args:
            obs: 动作前的观测
            action: 执行的动作
            reward: 获得的奖励
            next_obs: 动作后的观测
            done: 这手牌是否结束
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
