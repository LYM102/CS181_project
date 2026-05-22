# agents/sarsa_agent.py - SARSA 在线时序差分学习 Agent (占位)

from agents.base_agent import BaseAgent
from game.engine import Observation


class SARSAAgent(BaseAgent):
    """
    SARSA (State-Action-Reward-State-Action) 在线 on-policy Q 学习 Agent。

    核心公式:
      δ_t = R + γ·Q(s',a') - Q(s,a)
      Q(s,a) ← Q(s,a) + α·δ_t

    状态编码: s = (H_code, P_code, B_level, Pos)
    动作空间: A = {Fold(0), Call(1), Raise(2)}

    超参数:
      γ = 0.95, α = 0.1
      ε ← max(0.01, ε · 0.999)  (ε-greedy 探索衰减)

    TODO: 实现 Q-table、状态编码、SARSA 更新逻辑
    """

    def __init__(self, name: str = "SARSAAgent"):
        super().__init__(name=name)
        self.q_table = {}  # (state, action) → Q-value
        self.epsilon = 1.0
        self.alpha = 0.1
        self.gamma = 0.95

    def act(self, obs: Observation) -> int:
        # 占位: ε-greedy 策略
        import random
        if random.random() < self.epsilon:
            return random.choice(obs.legal_actions)
        # TODO: 从 Q-table 中选择最优动作
        return 1  # CALL (默认)

    def update(self, obs, action, reward, next_obs, done):
        # TODO: 实现 SARSA 更新
        self.epsilon = max(0.01, self.epsilon * 0.999)

    def _encode_state(self, obs: Observation) -> tuple:
        """
        将 Observation 编码为 Q-table 的状态键。

        s = (H_code, P_code, B_level, Pos)
        H_code: equity 离散化为 20 个 bin
        P_code: 公共牌强度编码
        B_level: 下注等级
        Pos: 位置
        """
        from game.evaluator import equity_to_bin
        h_code = equity_to_bin(obs.equity)
        p_code = len(obs.community_cards)  # 简化: 用公共牌数量表示阶段
        return (h_code, p_code, obs.betting_level, obs.position)
