# agents/expert_agent.py - 基于 OpenSpiel CFR 的纳什均衡专家策略 (占位)

from agents.base_agent import BaseAgent
from game.engine import Observation


class ExpertAgent(BaseAgent):
    """
    基于 OpenSpiel CFR 算法的纳什均衡基准 Agent。

    实现要点:
      - 使用 OpenSpiel 加载简化德州扑克游戏定义
      - 运行 CFR 迭代计算近似纳什均衡策略
      - 根据当前信息集采样动作

    TODO: 接入 OpenSpiel CFR 实现
    """

    def __init__(self, name: str = "ExpertAgent"):
        super().__init__(name=name)

    def act(self, obs: Observation) -> int:
        # 占位: 暂时使用简单启发式
        if obs.equity > 0.6 and RAISE_OBS(obs):
            return 2  # RAISE
        elif obs.equity > 0.3:
            return 1  # CALL
        else:
            return 0  # FOLD


def RAISE_OBS(obs: Observation) -> bool:
    """检查当前观测下 RAISE 是否为合法动作"""
    from game.constants import RAISE
    return RAISE in obs.legal_actions
