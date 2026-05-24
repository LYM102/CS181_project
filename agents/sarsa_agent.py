# agents/sarsa_agent.py - Q-learning off-policy temporal-difference learning Agent

import random
from collections import defaultdict

from agents.base_agent import BaseAgent
from game.engine import Observation


class QLearningAgent(BaseAgent):
    """
    Q-learning off-policy TD control Agent.

    Core update rule (off-policy, uses greedy max over next state):
      Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a'(Q(s',a')) - Q(s,a)]

    Key difference from SARSA:
      - SARSA uses the *actual* next action a' chosen by the policy (on-policy).
      - Q-learning uses the *greedy* max over all actions in s' (off-policy),
        making it converge to the optimal policy regardless of exploration.

    State encoding: s = (H_code, P_code, B_level, Pot_bin, Pos)
    Action space: A = {Fold(0), Call(1), Raise(2)}

    Hyperparameters:
      gamma = 0.95        (discount factor)
      alpha = 0.1         (learning rate)
      epsilon = 1.0 -> 0.01  (epsilon-greedy exploration with decay 0.999)
    """

    # Full action space for Q-table initialization
    ACTION_SPACE = [0, 1, 2]  # Fold, Call, Raise

    def __init__(self, name: str = "QLearningAgent"):
        super().__init__(name=name)
        # Q-table: state -> {action: q_value}
        self.q_table: dict[tuple, dict[int, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.ACTION_SPACE}
        )
        self.epsilon = 1.0   # exploration rate (decays over time)
        self.alpha = 0.1     # learning rate
        self.gamma = 0.95    # discount factor

    def act(self, obs: Observation) -> int:
        """
        Select action using epsilon-greedy policy.

        With probability epsilon: explore (random legal action).
        With probability 1-epsilon: exploit (greedy action from Q-table,
        restricted to legal actions).
        """
        state = self._encode_state(obs)

        # Exploration: random legal action
        if random.random() < self.epsilon:
            return random.choice(obs.legal_actions)

        # Exploitation: pick action with highest Q-value among legal actions
        q_values = self.q_table[state]
        legal_q = {a: q_values[a] for a in obs.legal_actions}
        max_q = max(legal_q.values())
        # Break ties randomly
        best_actions = [a for a, q in legal_q.items() if q == max_q]
        return random.choice(best_actions)

    def update(self, obs: Observation, action: int, reward: float,
               next_obs: Observation, done: bool) -> None:
        """
        Q-learning TD update (off-policy).

        Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a'(Q(s',a')) - Q(s,a)]

        When done=True (terminal state), the target simplifies to just R
        since there is no future reward.
        """
        state = self._encode_state(obs)
        q_current = self.q_table[state][action]

        if done:
            # Terminal state: no future reward
            td_target = reward
        else:
            # Off-policy: use max Q-value of next state (greedy over all actions)
            next_state = self._encode_state(next_obs)
            max_next_q = max(self.q_table[next_state].values())
            td_target = reward + self.gamma * max_next_q

        # TD error and Q-value update
        td_error = td_target - q_current
        self.q_table[state][action] = q_current + self.alpha * td_error

        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.999)

    def _encode_state(self, obs: Observation) -> tuple:
        """
        Encode Observation into a compact Q-table state key.

        s = (H_code, P_code, B_level, Pot_bin, Pos)
        H_code: equity discretized into 20 bins
        P_code: community card count (stage indicator: 0/3/4/5)
        B_level: current betting level
        Pot_bin: pot size discretized into 6 bins
        Pos: player position
        """
        from game.evaluator import equity_to_bin, pot_to_bin
        h_code = equity_to_bin(obs.equity)
        # Use community card count as stage indicator
        p_code = len(obs.community_cards)
        pot_bin = pot_to_bin(obs.pot)
        return (h_code, p_code, obs.betting_level, pot_bin, obs.position)

    def get_q_table_size(self) -> int:
        """Return the number of states visited in the Q-table."""
        return len(self.q_table)

    def reset_exploration(self, epsilon: float = 1.0) -> None:
        """Reset exploration rate (useful for retraining phases)."""
        self.epsilon = epsilon
