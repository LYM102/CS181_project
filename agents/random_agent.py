# agents/random_agent.py - 随机策略 Agent (用于测试和基准)

import random
from agents.base_agent import BaseAgent
from game.engine import Observation
from game.constants import CALL, RAISE


class RandomAgent(BaseAgent):
    """随机选择合法动作的 Agent，用于测试和基准对照"""

    def __init__(self, name: str = "RandomAgent", fold_prob: float = 0.1):
        """
        Args:
            name: Agent 名称
            fold_prob: 弃牌概率 (仅在 FOLD 为合法动作时)
        """
        super().__init__(name=name)
        self.fold_prob = fold_prob

    def act(self, obs: Observation) -> int:
        legal = obs.legal_actions
        # 简单策略: 以一定概率弃牌，否则在 CALL/RAISE 中随机
        if CALL in legal and RAISE in legal:
            if random.random() < self.fold_prob and 0 in legal:
                return 0  # FOLD
            return random.choice([CALL, RAISE])
        # 只能 CALL 的情况 (已达加注上限)
        return CALL if CALL in legal else legal[0]
