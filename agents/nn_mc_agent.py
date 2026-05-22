# agents/nn_mc_agent.py - BNN (贝叶斯神经网络) + MC Q 表 Agent (占位)

from agents.base_agent import BaseAgent
from game.engine import Observation


class NN_MCAgent(BaseAgent):
    """
    贝叶斯神经网络-蒙特卡洛 Agent。

    核心思路:
      用 BNN (MC Dropout) 替代离散贝叶斯推断，可捕获多步动作模式
      (如 flop check → turn raise 诈唬签名)。

    BNN 输入:
      - Own hand equity: 标量 [0,1]
      - Community card strength: 标量 [0,1]
      - Betting round: one-hot (4 dims)
      - Opponent action matrix M_opp: shape R×k (R=4轮, k=4动作槽)
      - Own action matrix M_self: shape R×k
      - New-community-card flag: binary

    BNN 输出:
      - 对手手牌强度分布 → argmax → O_NN

    MC Q-table 状态:
      s = (S, B, O_NN)
      S: equity 离散 20 bins
      B: 下注等级 {0,1,2,3}
      O_NN: BNN 预测的对手信念标签

    TODO: 实现 BNN 网络结构、MC Dropout 推理、Q-table 更新
    """

    def __init__(self, name: str = "NN_MCAgent"):
        super().__init__(name=name)
        self.q_table = {}
        self.bnn_model = None  # TODO: 初始化 BNN

    def act(self, obs: Observation) -> int:
        # 占位
        from game.constants import CALL
        return CALL

    def _encode_action_matrix(self, action_history: list) -> list:
        """
        将动作历史编码为 R×k 矩阵。

        Args:
            action_history: 按轮次组织的动作序列

        Returns:
            R×k 矩阵 (R=4, k=4)
        """
        R, k = 4, 4
        matrix = [[-1] * k for _ in range(R)]
        # TODO: 填充实际动作
        return matrix

    def _predict_opponent_strength(self, obs: Observation) -> str:
        """
        使用 BNN 预测对手手牌强度标签。

        Returns:
            "strong" / "mid" / "weak"
        """
        # TODO: 实现 MC Dropout 推理
        return "mid"
