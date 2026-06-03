# agents/nfsp_agent.py — Neural Fictitious Self-Play Agent (52-card)
#
# Reference: Heinrich & Silver (2016) "Deep Reinforcement Learning from
#            Self-Play in Imperfect-Information Games"
#
# Architecture:
#   - Q-network (DQN): learns best response via RL (experience replay)
#   - Policy network: learns average strategy via supervised learning (reservoir)
#   - Self-play with anticipatory parameter η mixes both strategies
#   - Evaluation uses only the average policy (mixed strategy → Nash)
#
# Compatible with standard 52-card Texas Hold'em:
#   - 52 cards (4 suits × 13 ranks: 2-A)
#   - Betting levels [10, 20, 40, 80, 160, 320], max 4 raises/round
#   - Actions: Fold(0), Call(1), Raise(2)

from __future__ import annotations
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base_agent import BaseAgent
from game.engine import Observation
from game.constants import FOLD, CALL, RAISE, BETTING_LEVELS


# =========================================================================
#  Neural Network Definitions
# =========================================================================

class DQN(nn.Module):
    """Deep Q-Network: state → Q-values for each action."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Average policy network: state → action probabilities."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Return action probabilities (softmax)."""
        return F.softmax(self.net(x), dim=-1)


# =========================================================================
#  Replay Buffers
# =========================================================================

class CircularBuffer:
    """Fixed-size circular replay buffer for DQN (RL memory)."""

    def __init__(self, capacity: int = 200_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class ReservoirBuffer:
    """Reservoir sampling buffer for supervised learning (SL memory).

    Guarantees uniform sampling over the entire training history,
    which is critical for NFSP's average policy convergence.
    """

    def __init__(self, capacity: int = 2_000_000):
        self.buffer = []
        self.capacity = capacity
        self.total_seen = 0

    def add(self, state, action):
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action))
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (state, action)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================================
#  NFSP Agent
# =========================================================================

class NFSPAgent(BaseAgent):
    """
    Neural Fictitious Self-Play Agent for standard 52-card Texas Hold'em.

    Training mode:
      - With prob η: use DQN ε-greedy (best response) → store in M_RL
      - With prob 1-η: use Policy network (average strategy)
      - Always: store (s, a) in M_SL (reservoir sampling)

    Evaluation mode:
      - Use only Policy network (average/mixed strategy)
      - Sample action from probability distribution

    This converges to approximate Nash equilibrium via self-play.
    """

    INPUT_DIM = 15
    OUTPUT_DIM = 3  # Fold, Call, Raise

    def __init__(self, name: str = "NFSPAgent",
                 hidden_dim: int = 128,
                 eta: float = 0.1,
                 epsilon: float = 0.06,
                 gamma: float = 1.0,
                 q_lr: float = 0.01,
                 policy_lr: float = 0.005,
                 batch_size: int = 128,
                 rl_buffer_size: int = 200_000,
                 sl_buffer_size: int = 2_000_000,
                 target_update_freq: int = 1000,
                 device: str = "cpu",
                 load_model_path: str = None):
        super().__init__(name=name)

        self.eta = eta
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.train_mode = False

        # Networks (larger hidden dim for 52-card complexity)
        self.q_network = DQN(self.INPUT_DIM, hidden_dim, self.OUTPUT_DIM).to(device)
        self.target_network = DQN(self.INPUT_DIM, hidden_dim, self.OUTPUT_DIM).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.policy_network = PolicyNetwork(self.INPUT_DIM, hidden_dim, self.OUTPUT_DIM).to(device)

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)

        # Buffers
        self.rl_buffer = CircularBuffer(capacity=rl_buffer_size)
        self.sl_buffer = ReservoirBuffer(capacity=sl_buffer_size)

        # Training counters
        self.train_steps = 0
        self._current_mode = 'average'

        if load_model_path:
            self.load_model(load_model_path)

    # ==================================================================
    #  State Encoding (15-dim feature vector)
    # ==================================================================

    def _encode_state(self, obs: Observation) -> np.ndarray:
        """
        Encode observation into 15-dim normalized feature vector.

        All features are derived from our game's Observation object.
        No external poker library calls — only uses obs fields directly.

        Features:
          [0]     equity             : hand win rate [0,1] (from obs.equity)
          [1]     pot_ratio          : pot / (pot + player_chips)
          [2]     pot_odds           : call_amount / (pot + call_amount)
          [3]     SPR                : eff_stack / max(pot,1), clipped [0,1]
          [4:8]   round_onehot       : preflop/flop/turn/river
          [8]     betting_level_norm : betting_level / max_level
          [9]     raises_norm        : raises_this_round / max_raises
          [10]    position           : dealer(0) or not(1)
          [11]    chips_ratio        : own_chips / total_chips
          [12]    can_raise          : 1 if RAISE in legal_actions
          [13]    call_amount_norm   : call_needed / max_bet_level
          [14]    community_stage    : len(community_cards) / 5.0
        """
        features = np.zeros(self.INPUT_DIM, dtype=np.float32)

        max_level = len(BETTING_LEVELS) - 1  # 5 for 52-card
        max_bet = BETTING_LEVELS[-1]  # 320

        # [0] Equity
        features[0] = obs.equity

        # [1] Pot ratio
        total = obs.pot + obs.player_chips
        features[1] = obs.pot / total if total > 0 else 0.0

        # [2] Pot odds
        call_amount = max(0, obs.current_bet - obs.player_round_bet)
        pot_plus_call = obs.pot + call_amount
        features[2] = call_amount / pot_plus_call if pot_plus_call > 0 else 0.0

        # [3] SPR (stack-to-pot ratio), normalized
        eff_stack = min(obs.player_chips, obs.opponent_chips)
        spr = eff_stack / max(obs.pot, 1)
        features[3] = min(spr / 20.0, 1.0)

        # [4:8] Round one-hot (PREFLOP=0, FLOP=1, TURN=2, RIVER=3)
        round_idx = min(obs.current_round, 3)
        features[4 + round_idx] = 1.0

        # [8] Betting level normalized
        features[8] = obs.betting_level / max(max_level, 1)

        # [9] Raises this round normalized
        features[9] = obs.raises_this_round / 4.0  # MAX_RAISES=4

        # [10] Position (0=dealer/SB, 1=BB)
        features[10] = float(obs.position)

        # [11] Chips ratio
        total_chips = obs.player_chips + obs.opponent_chips
        features[11] = obs.player_chips / total_chips if total_chips > 0 else 0.5

        # [12] Can raise flag
        features[12] = 1.0 if RAISE in obs.legal_actions else 0.0

        # [13] Call amount normalized by max bet level
        features[13] = min(call_amount / max_bet, 1.0)

        # [14] Community card stage (0/3/4/5 → 0/0.6/0.8/1.0)
        features[14] = len(obs.community_cards) / 5.0

        return features

    # ==================================================================
    #  Action Selection
    # ==================================================================

    def act(self, obs: Observation) -> int:
        """
        Select action using average policy (for evaluation).
        Samples from policy network's probability distribution.
        Ensures only legal actions are selected.
        """
        return self._act_policy(obs)

    def act_train(self, obs: Observation) -> int:
        """
        Training-time action selection using η-anticipatory dynamics.

        With prob η: use DQN ε-greedy (best response)
        With prob 1-η: use Policy network (average strategy)
        """
        if random.random() < self.eta:
            self._current_mode = 'best_response'
            return self._act_dqn(obs)
        else:
            self._current_mode = 'average'
            return self._act_policy(obs)

    def _act_dqn(self, obs: Observation) -> int:
        """ε-greedy action selection from Q-network."""
        legal_actions = obs.legal_actions
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        state = self._encode_state(obs)
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(x).squeeze(0)

        # Mask illegal actions with -inf
        for a in range(self.OUTPUT_DIM):
            if a not in legal_actions:
                q_values[a] = float('-inf')

        return int(q_values.argmax().item())

    def _act_policy(self, obs: Observation) -> int:
        """Sample action from policy network (mixed strategy)."""
        legal_actions = obs.legal_actions
        state = self._encode_state(obs)
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_network(x).squeeze(0).cpu().numpy()

        # Mask illegal actions and renormalize
        masked_probs = np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        for a in legal_actions:
            masked_probs[a] = probs[a]

        prob_sum = masked_probs.sum()
        if prob_sum > 1e-8:
            masked_probs /= prob_sum
        else:
            # Fallback: uniform over legal actions
            for a in legal_actions:
                masked_probs[a] = 1.0 / len(legal_actions)

        return int(np.random.choice(self.OUTPUT_DIM, p=masked_probs))

    # ==================================================================
    #  Training Methods
    # ==================================================================

    def store_rl_transition(self, state: np.ndarray, action: int,
                            reward: float, next_state: np.ndarray, done: bool):
        """Store transition in RL replay buffer (only for best-response steps)."""
        self.rl_buffer.add(state, action, reward, next_state, done)

    def store_sl_transition(self, state: np.ndarray, action: int):
        """Store (state, action) in SL reservoir buffer (all steps)."""
        self.sl_buffer.add(state, action)

    def train_q_network(self) -> float:
        """
        Train DQN on a mini-batch from RL buffer.
        Returns loss value for logging.
        """
        if len(self.rl_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.rl_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q-values
        q_values = self.q_network(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (from target network)
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(q_selected, targets)

        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train_policy_network(self) -> float:
        """
        Train policy network on a mini-batch from SL buffer.
        Supervised learning: cross-entropy loss on (state, action) pairs.
        Returns loss value for logging.
        """
        if len(self.sl_buffer) < self.batch_size:
            return 0.0

        states, actions = self.sl_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)

        # Policy network outputs probabilities (softmax already applied)
        probs = self.policy_network(states_t)
        # Cross-entropy: -log(prob of correct action)
        log_probs = torch.log(probs + 1e-8)
        loss = F.nll_loss(log_probs, actions_t)

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        return loss.item()

    # ==================================================================
    #  Persistence
    # ==================================================================

    def save_model(self, filepath: str) -> None:
        """Save both networks' weights."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "policy_network": self.policy_network.state_dict(),
            "train_steps": self.train_steps,
        }, filepath)
        print(f"[NFSPAgent] Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load both networks' weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.train_steps = checkpoint.get("train_steps", 0)
        print(f"[NFSPAgent] Model loaded from {filepath} (steps={self.train_steps})")

    def reset(self) -> None:
        """No per-hand state to reset (stateless feature encoding)."""
        pass
