"""Tabular SARSA agent for abstracted heads-up Texas Hold'em."""

import random
from collections import defaultdict

from agents.base_agent import BaseAgent
from game.engine import Observation
from game.evaluator import equity_to_bin, pot_to_bin


class SarsaAgent(BaseAgent):
    """On-policy TD control with ε-greedy exploration over abstract states."""

    ACTION_SPACE = [0, 1, 2]  # Fold, Call, Raise

    def __init__(self, name: str = "SarsaAgent",
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.999,
                 epsilon_min: float = 0.01,
                 load_q_table_path: str = None):
        super().__init__(name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table: dict[tuple, list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0])

        if load_q_table_path:
            self.load_q_table(load_q_table_path)

    def act(self, obs: Observation) -> int:
        """ε-greedy action selection."""
        state = self._encode_state(obs)
        legal_actions = obs.legal_actions

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploit: best Q among legal actions
        q_vals = self.q_table[state]
        best_value = max(q_vals[a] for a in legal_actions)
        best_actions = [a for a in legal_actions if q_vals[a] == best_value]
        return random.choice(best_actions)

    def update(self, obs, action, reward, next_obs, done) -> None:
        """No-op: SARSA requires next_action. Use learn() in training loops."""
        pass

    def learn(self, state: tuple, action: int, reward: float,
              next_state: tuple, next_action: int, done: bool) -> None:
        """Core SARSA update: Q(s,a) += α * [r + γ*Q(s',a') - Q(s,a)]."""
        q_current = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            q_next = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * q_next

        td_error = td_target - q_current
        self.q_table[state][action] += self.alpha * td_error

    def _encode_state(self, obs: Observation) -> tuple:
        """Encode observation into (H_code, P_code, B_level, Pot_bin, Pos)."""
        h_code = equity_to_bin(obs.equity, bins=20)
        p_code = len(obs.community_cards)
        pot_bin = pot_to_bin(obs.pot)
        return (h_code, p_code, obs.betting_level, pot_bin, obs.position)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_size(self) -> int:
        return len(self.q_table)
    
    def save_q_table(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filepath} (size={len(self.q_table)})")

    def load_q_table(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        self.q_table.clear()
        for state, q_vals in loaded.items():
            self.q_table[state] = q_vals
        print(f"Q-table loaded from {filepath} (size={len(self.q_table)})")