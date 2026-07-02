# agents/sarsa_agent.py - SARSA (on-policy TD control) Agent

import random
from collections import defaultdict

from agents.base_agent import BaseAgent
from game.engine import Observation
from game.evaluator import equity_to_bin, pot_to_bin


class SarsaAgent(BaseAgent):
    """
    SARSA – On‑policy Temporal Difference Control for Texas Hold'em.

    Core update rule (on‑policy):
        Q(s,a) <- Q(s,a) + α * [ R + γ * Q(s', a') - Q(s,a) ]

    where a' is the *actual* next action chosen by the current policy.
    This makes SARSA learn the value of the policy it follows, which is
    more stable in non‑stationary environments (e.g., self‑play).

    State encoding (same as Q‑learning agent):
        s = (H_code, P_code, B_level, Pot_bin, Pos)

    Hyperparameters:
        α = 0.1          # learning rate
        γ = 0.95         # discount factor
        ε = 1.0 → 0.01   # exploration rate (decays by 0.999 each hand)
    """

    ACTION_SPACE = [0, 1, 2]   # Fold, Call, Raise

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

        # Q‑table: state (tuple) -> list of Q-values [Q_fold, Q_call, Q_raise]
        self.q_table: dict[tuple, list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0]

            
        )

        if load_q_table_path:
            self.load_q_table(load_q_table_path)


    #  Public interface (required by BaseAgent)

    def act(self, obs: Observation) -> int:
        """
        ε‑greedy action selection.
        Uses the current Q‑table and epsilon.
        """
        state = self._encode_state(obs)
        legal_actions = obs.legal_actions

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation: choose action with highest Q among legal actions
        q_vals = self.q_table[state]
        best_value = max(q_vals[a] for a in legal_actions)
        best_actions = [a for a in legal_actions if q_vals[a] == best_value]
        return random.choice(best_actions)

    def update(self, obs: Observation, action: int, reward: float,
               next_obs: Observation, done: bool) -> None:
        """
        SARSA update – but note: BaseAgent.update() does NOT provide `next_action`.
        Therefore this method cannot perform a proper SARSA update.

        To use SARSA, please call `learn()` directly in your training loop
        (see example in class docstring or training section below).

        If you still call this method, it will behave as a no‑op.
        """
        # No‑op: SARSA requires next_action, which is not available here.
        pass

    #  SARSA‑specific learning method (to be used in training loops)

    def learn(self, state: tuple, action: int, reward: float,
              next_state: tuple, next_action: int, done: bool) -> None:
        """
        The core SARSA update. Call this after each environment step.

        Args:
            state      : encoded state before action
            action     : action taken
            reward     : immediate reward (usually 0, nonzero only at terminal)
            next_state : encoded state after action
            next_action: action that will be taken in next_state (from current policy)
            done       : whether the episode (hand) is finished
        """
        q_current = self.q_table[state][action]

        if done:
            # Terminal state: no future reward
            td_target = reward
        else:
            q_next = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * q_next

        # TD error and update
        td_error = td_target - q_current
        self.q_table[state][action] += self.alpha * td_error

    #  Helper methods

    def _encode_state(self, obs: Observation) -> tuple:
        """
        Convert Observation into a compact tuple key for the Q‑table.

        Components:
            H_code     : equity discretized into 20 bins (0..19)
            P_code     : number of community cards (0,3,4,5)
            B_level    : current betting level (0..3)
            Pot_bin    : pot size discretized into 6 bins (0..5)
            Pos        : player position (0 or 1)
        """
        h_code = equity_to_bin(obs.equity, bins=20)
        p_code = len(obs.community_cards)          # stage indicator
        pot_bin = pot_to_bin(obs.pot)
        return (h_code, p_code, obs.betting_level, pot_bin, obs.position)

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each hand (episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_exploration(self, epsilon: float = 1.0) -> None:
        """Reset epsilon to a given value (useful for restarting training)."""
        self.epsilon = epsilon

    def get_q_table_size(self) -> int:
        """Return number of distinct states visited so far."""
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


#  How to use SarsaAgent in a training loop (example)
#
# from game.engine import GameEngine
# from agents.sarsa_agent import SarsaAgent
#
# agent = SarsaAgent()
# opponent = RandomAgent()   # or any fixed agent
# env = GameEngine(agent, opponent)
#
# for episode in range(num_hands):
#     obs = env.reset_hand()
#     state = agent._encode_state(obs)
#     action = agent.act(obs)
#     done = False
#
#     while not done:
#         next_obs, reward, done, _ = env.step(action)
#         if done:
#             # Terminal update (no next_action)
#             agent.learn(state, action, reward, None, None, done=True)
#             break
#
#         # Choose next action using the current policy
#         next_action = agent.act(next_obs)
#         next_state = agent._encode_state(next_obs)
#
#         # SARSA update for the transition (s,a,r,s',a')
#         agent.learn(state, action, reward, next_state, next_action, done=False)
#
#         # Move to next step
#         state, action = next_state, next_action
#
#     # Decay epsilon after each hand
#     agent.decay_epsilon()
#