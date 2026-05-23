# agents/sarsa_agent.py - SARSA online temporal-difference learning Agent (placeholder)

from agents.base_agent import BaseAgent
from game.engine import Observation


class SARSAAgent(BaseAgent):
    """
    SARSA (State-Action-Reward-State-Action) online on-policy Q-learning Agent.

    Core formula:
      δ_t = R + γ·Q(s',a') - Q(s,a)
      Q(s,a) ← Q(s,a) + α·δ_t

    State encoding: s = (H_code, P_code, B_level, Pos)
    Action space: A = {Fold(0), Call(1), Raise(2)}

    Hyperparameters:
      γ = 0.95, α = 0.1
      ε ← max(0.01, ε · 0.999)  (ε-greedy exploration decay)

    TODO: Implement Q-table, state encoding, SARSA update logic
    """

    def __init__(self, name: str = "SARSAAgent"):
        super().__init__(name=name)
        self.q_table = {}  # (state, action) → Q-value
        self.epsilon = 1.0
        self.alpha = 0.1
        self.gamma = 0.95

    def act(self, obs: Observation) -> int:
        # Placeholder: ε-greedy policy
        import random
        if random.random() < self.epsilon:
            return random.choice(obs.legal_actions)
        # TODO: Select optimal action from Q-table
        return 1  # CALL (default)

    def update(self, obs, action, reward, next_obs, done):
        # TODO: Implement SARSA update
        self.epsilon = max(0.01, self.epsilon * 0.999)

    def _encode_state(self, obs: Observation) -> tuple:
        """
        Encode Observation into a Q-table state key.

        s = (H_code, P_code, B_level, Pot_bin, Pos)
        H_code: equity discretized into 20 bins
        P_code: community card strength encoding
        B_level: betting level
        Pot_bin: pot size discretized into 6 bins
        Pos: position
        """
        from game.evaluator import equity_to_bin, pot_to_bin
        h_code = equity_to_bin(obs.equity)
        # Simplified: use community card count to represent stage
        p_code = len(obs.community_cards)
        pot_bin = pot_to_bin(obs.pot)
        return (h_code, p_code, obs.betting_level, pot_bin, obs.position)
