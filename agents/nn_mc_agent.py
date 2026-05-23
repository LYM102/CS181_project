# agents/nn_mc_agent.py - BNN (Bayesian Neural Network) + MC Q-table Agent (placeholder)

from agents.base_agent import BaseAgent
from game.engine import Observation


class NN_MCAgent(BaseAgent):
    """
    Bayesian Neural Network - Monte Carlo Agent.

    Core idea:
      Use BNN (MC Dropout) to replace discrete Bayesian inference,
      capable of capturing multi-step action patterns
      (e.g., flop check → turn raise bluff signature).

    BNN input:
      - Own hand equity: scalar [0,1]
      - Community card strength: scalar [0,1]
      - Betting round: one-hot (4 dims)
      - Opponent action matrix M_opp: shape R×k (R=4 rounds, k=4 action slots)
      - Own action matrix M_self: shape R×k
      - New-community-card flag: binary

    BNN output:
      - Opponent hand strength distribution → argmax → O_NN

    MC Q-table state:
      s = (S, B, O_NN)
      S: equity discretized into 20 bins
      B: betting level {0,1,2,3}
      O_NN: BNN predicted opponent belief label

    TODO: Implement BNN network structure, MC Dropout inference, Q-table update
    """

    def __init__(self, name: str = "NN_MCAgent"):
        super().__init__(name=name)
        self.q_table = {}
        self.bnn_model = None  # TODO: Initialize BNN

    def act(self, obs: Observation) -> int:
        # Placeholder
        from game.constants import CALL
        return CALL

    def _encode_action_matrix(self, action_history: list) -> list:
        """
        Encode action history into an R×k matrix.

        Args:
            action_history: action sequence organized by round

        Returns:
            R×k matrix (R=4, k=4)
        """
        R, k = 4, 4
        matrix = [[-1] * k for _ in range(R)]
        # TODO: Fill in actual actions
        return matrix

    def _predict_opponent_strength(self, obs: Observation) -> str:
        """
        Predict opponent hand strength label using BNN.

        Returns:
            "strong" / "mid" / "weak"
        """
        # TODO: Implement MC Dropout inference
        return "mid"
