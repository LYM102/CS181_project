# agents/nn_mc_agent.py - BNN (Bayesian Neural Network) + MC Q-table Agent (52-card)

from __future__ import annotations
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.base_agent import BaseAgent
from game.engine import Observation
from game.evaluator import equity_to_bin, pot_to_bin, compute_equity

# =========================================================================
#  BNN Model with MC Dropout
# =========================================================================

class BNNWithMCDropout(nn.Module):
    """
    Bayesian Neural Network using MC Dropout (Gal & Ghahramani, 2016).

    Dropout layers remain active during both training AND inference,
    enabling uncertainty estimation via multiple stochastic forward passes.

    Architecture:
        Input(42) → Dense(128) + ReLU + Dropout(0.15)
                 → Dense(64)  + ReLU + Dropout(0.15)
                 → Dense(32)  + ReLU + Dropout(0.15)
                 → Dense(3) + Softmax

        Features [0:39]  = public info (same as inference)
        Features [39:42] = opponent hand (equity, avg_rank, suited) — masked at inference
    """

    def __init__(self, input_dim=42, hidden_dims=(128, 64, 32),
                 num_classes=3, dropout_rate=0.15):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        """Single forward pass (logits). Softmax applied by caller."""
        return self.net(x)

    def predict_proba(self, x, mc_samples=20):
        """
        MC Dropout inference: run T forward passes (dropout ON).

        Args:
            x: input tensor (batch_size, input_dim)
            mc_samples: number of MC samples

        Returns:
            mean_probs: shape (batch_size, num_classes) — mean probability
            uncertainty: shape (batch_size,) — std of winning class
        """
        self.train()  # keep dropout active during inference
        all_probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.stack(all_probs, axis=0)  # (T, batch, num_classes)
        mean_probs = all_probs.mean(axis=0)       # (batch, num_classes)
        # uncertainty = std of the predicted class probability
        pred_classes = mean_probs.argmax(axis=1)   # (batch,)
        batch_indices = np.arange(len(pred_classes))
        uncertainty = all_probs[:, batch_indices, pred_classes].std(axis=0)  # (batch,)
        return mean_probs, uncertainty


# =========================================================================
#  NN_MCAgent
# =========================================================================

class NN_MCAgent(BaseAgent):
    """
    Bayesian Neural Network + SARSA TD-Learning Agent.

    Core pipeline:
      1. BNN (MC Dropout) predicts opponent hand strength probability
         distribution from enriched features including board texture,
         pot odds, SPR, and multi-round action patterns.
      2. SARSA Q-table uses (equity_bin, B_level, Pot_bin,
         O_weak, O_mid, O_strong) as state, updated via TD bootstrapping.

    Adapted for standard 52-card Texas Hold'em.

    References:
      - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
      - Sutton & Barto: Reinforcement Learning (SARSA, TD methods)
    """

    ACTION_SPACE = [0, 1, 2]  # Fold, Call, Raise
    OPP_STRENGTH_LABELS = {"weak": 0, "mid": 1, "strong": 2}
    LABEL_TO_STR = {0: "weak", 1: "mid", 2: "strong"}
    BNN_FEATURE_DIM = 47  # enriched feature vector dimension

    # State encoding modes:
    #   'prob3': full 3-class probability bins (3×3×3=27 BNN states) → 4860 total
    #   'compact': argmax + confidence (3×2=6 BNN states) → 1080 total
    #   'argmax': just argmax class (3 BNN states) → 540 total
    VALID_STATE_MODES = ('prob3', 'compact', 'argmax')

    def __init__(self, name: str = "NN_MCAgent",
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.05,
                 alpha: float = 0.1,          # Q-table learning rate
                 gamma: float = 0.95,          # discount factor
                 mc_samples: int = 20,         # MC Dropout samples
                 device: str = "cpu",
                 player_id: int = 0,           # which player this agent represents
                 load_model_path: str = None,
                 state_mode: str = 'compact',  # state encoding mode
                 bnn_hidden_dims: tuple = (128, 64, 32),
                 bnn_dropout: float = 0.15):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma
        self.mc_samples = mc_samples
        self.device = device
        self.player_id = player_id
        self.state_mode = state_mode
        assert state_mode in self.VALID_STATE_MODES, f"Invalid state_mode: {state_mode}"

        # --- BNN model ---
        self.bnn_model = BNNWithMCDropout(
            input_dim=self.BNN_FEATURE_DIM,
            hidden_dims=bnn_hidden_dims,
            dropout_rate=bnn_dropout,
        ).to(self.device)
        self.bnn_trained = False

        # --- MC Q-table: state tuple → [Q_fold, Q_call, Q_raise] ---
        self.q_table: dict[tuple, list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0]
        )

        # --- Per-hand action history (external recording) ---
        self._opp_actions: list[tuple[int, int]] = []   # (round, action)
        self._self_actions: list[tuple[int, int]] = []  # (round, action)
        self._prev_community_count: int = 0
        self._auto_record_self: bool = True     # auto-record own action in act()
                                                # set False when training loop
                                                # handles recording externally

        if load_model_path:
            self.load_model(load_model_path)

    # ==================================================================
    #  Public interface (BaseAgent)
    # ==================================================================

    def act(self, obs: Observation) -> int:
        """
        ε-greedy action selection using MC Q-table.

        The Q-table state includes O_NN (BNN-predicted opponent belief).
        If BNN is untrained, O_NN defaults to "mid".

        Auto-resets action history when a new hand is detected
        (community card count decreased from previous observation).
        Also auto-records own action for BNN feature building (evaluation mode).
        """
        # Auto-detect new hand: community card count dropped (new hand started)
        cc_count = len(obs.community_cards)
        if cc_count < self._prev_community_count:
            self.reset()
        self._prev_community_count = cc_count

        state = self._encode_state(obs)
        legal_actions = obs.legal_actions

        # ε-greedy
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            q_vals = self.q_table[state]
            best_value = max(q_vals[a] for a in legal_actions)
            best_actions = [a for a in legal_actions if q_vals[a] == best_value]
            action = random.choice(best_actions)

        # Auto-record own action for BNN feature building
        if self._auto_record_self:
            self._self_actions.append((obs.current_round, action))
        return action

    def reset(self) -> None:
        """Reset per-hand tracking state."""
        self._opp_actions = []
        self._self_actions = []
        self._prev_community_count = 0

    def update(self, obs: Observation, action: int, reward: float,
               next_obs: Observation, done: bool) -> None:
        """No-op: use learn_mc() in training loop for MC updates."""
        pass

    # ==================================================================
    #  External action recording (called by training loop)
    # ==================================================================

    def record_action(self, player: int, action: int, round_num: int) -> None:
        """
        Record an action taken by a player in the current round.
        Must be called by the training loop after each step.

        Args:
            player: player index (0 or 1)
            action: FOLD(0) / CALL(1) / RAISE(2)
            round_num: PREFLOP(0) / FLOP(1) / TURN(2) / RIVER(3)
        """
        if player == self.player_id:
            self._self_actions.append((round_num, action))
        else:
            self._opp_actions.append((round_num, action))

    # ==================================================================
    #  State encoding
    # ==================================================================

    def _encode_state(self, obs: Observation) -> tuple:
        """
        Build Q-table state key based on self.state_mode.

        Modes:
          'prob3':   (H_code, B_level, Pot_bin, O_weak, O_mid, O_strong)
                     Total: 20×3×3×3×3×3 = 4860 states
          'compact': (H_code, B_level, Pot_bin, opp_class, confidence)
                     Total: 20×3×3×3×2 = 1080 states
          'argmax':  (H_code, B_level, Pot_bin, opp_class)
                     Total: 20×3×3×3 = 540 states
        """
        h_code = equity_to_bin(obs.equity, num_bins=20)
        pot_bin_val = pot_to_bin(obs.pot)

        if self.bnn_trained:
            probs, uncertainty = self._predict_proba(obs)

            if self.state_mode == 'prob3':
                o_weak   = min(int(probs[0] * 3), 2)
                o_mid    = min(int(probs[1] * 3), 2)
                o_strong = min(int(probs[2] * 3), 2)
                return (h_code, obs.betting_level, pot_bin_val,
                        o_weak, o_mid, o_strong)

            elif self.state_mode == 'compact':
                opp_class = int(probs.argmax())  # 0=weak, 1=mid, 2=strong
                confidence = 1 if float(probs.max()) >= 0.5 else 0  # high/low
                return (h_code, obs.betting_level, pot_bin_val,
                        opp_class, confidence)

            else:  # 'argmax'
                opp_class = int(probs.argmax())
                return (h_code, obs.betting_level, pot_bin_val, opp_class)
        else:
            # Default when BNN is untrained
            if self.state_mode == 'prob3':
                return (h_code, obs.betting_level, pot_bin_val, 1, 1, 1)
            elif self.state_mode == 'compact':
                return (h_code, obs.betting_level, pot_bin_val, 1, 0)
            else:  # 'argmax'
                return (h_code, obs.betting_level, pot_bin_val, 1)

    # ==================================================================
    #  BNN feature encoding
    # ==================================================================

    def _encode_bnn_features(self, obs: Observation,
                             opp_equity: float = None,
                             opp_rank_avg: float = None,
                             opp_suited: float = None) -> np.ndarray:
        """
        Build the 47-dim BNN input feature vector.

        Features [0:44]  — always available (public info):
          [0]    own_equity              : float [0,1]
          [1]    board_paired            : binary {0,1}
          [2]    flush_draw_possible     : binary {0,1}
          [3]    straight_draw_possible  : binary {0,1}
          [4]    board_connectivity      : float [0,1]
          [5:9]  betting_round_onehot    : 4 dims
          [9:25] opponent_action_matrix  : 16 dims (flattened 4×4, values +1 shifted)
          [25:41] self_action_matrix     : 16 dims (flattened 4×4, values +1 shifted)
          [41]   new_community_flag      : binary {0,1}
          [42]   pot_odds                : float [0,1]
          [43]   SPR                     : float [0,1] (clipped)

        Features [44:47] — opponent hand (MASK=0.5 at inference):
          [44]   opp_equity   : [0,1] or 0.5
          [45]   opp_rank_avg : [0,1] or 0.5
          [46]   opp_suited   : {0,1} or 0.5

        Training: 50% random mask on opponent features.
        Inference: ALWAYS masked (opponent hand unknown).
        """
        features = np.zeros(self.BNN_FEATURE_DIM, dtype=np.float32)

        # --- Own hand equity ---
        features[0] = obs.equity

        # --- Board texture (requires treys) ---
        from treys import Card
        cc = obs.community_cards
        if len(cc) >= 3:
            ranks = [Card.get_rank_int(c) for c in cc]
            suits = [Card.get_suit_int(c) for c in cc]
            # Paired board
            features[1] = 1.0 if len(set(ranks)) < len(ranks) else 0.0
            # Flush draw: 2+ cards of same suit (but less than 5)
            max_suit_count = max(suits.count(s) for s in set(suits))
            features[2] = 1.0 if max_suit_count >= 2 else 0.0
            # Straight draw: check if 3+ ranks within a span of 5
            unique_ranks = sorted(set(ranks))
            has_straight_draw = False
            for i in range(len(unique_ranks) - 2):
                if unique_ranks[i + 2] - unique_ranks[i] <= 5:
                    has_straight_draw = True
                    break
            features[3] = 1.0 if has_straight_draw else 0.0
            # Board connectivity: 1 - normalized avg rank gap
            if len(unique_ranks) >= 2:
                gaps = [unique_ranks[i+1] - unique_ranks[i] for i in range(len(unique_ranks)-1)]
                avg_gap = sum(gaps) / len(gaps)
                features[4] = 1.0 - min(avg_gap / 6.0, 1.0)  # normalize to [0,1]

        # --- Betting round one-hot ---
        round_idx = min(obs.current_round, 3)
        features[5 + round_idx] = 1.0

        # --- Action matrices (values +1 shifted: padding=0, FOLD=1, CALL=2, RAISE=3) ---
        opp_matrix = self._encode_action_matrix(self._opp_actions)
        for i in range(4):
            for j in range(4):
                features[9 + i * 4 + j] = opp_matrix[i][j] + 1.0  # shift -1→0, 0→1, 1→2, 2→3

        self_matrix = self._encode_action_matrix(self._self_actions)
        for i in range(4):
            for j in range(4):
                features[25 + i * 4 + j] = self_matrix[i][j] + 1.0

        # --- New community card flag ---
        current_cc = len(obs.community_cards)
        features[41] = 1.0 if current_cc > self._prev_community_count else 0.0

        # --- Pot odds = call_amount / (pot + call_amount) ---
        call_amount = obs.current_bet - obs.player_round_bet
        total_after_call = obs.pot + call_amount
        if total_after_call > 0:
            features[42] = min(call_amount / total_after_call, 1.0)

        # --- SPR = effective_stack / pot ---
        eff_stack = min(obs.player_chips, obs.opponent_chips)
        if obs.pot > 0:
            spr = eff_stack / obs.pot
            features[43] = min(spr / 20.0, 1.0)  # clip at SPR=20

        # --- Opponent hand features (masked at inference) ---
        features[44] = 0.5 if opp_equity is None else opp_equity
        features[45] = 0.5 if opp_rank_avg is None else opp_rank_avg
        features[46] = 0.5 if opp_suited is None else opp_suited

        return features

    def _encode_action_matrix(self, action_history: list) -> list:
        """
        Encode action history into a 4×4 matrix (R=4 rounds, k=4 max actions per round).

        Each cell: FOLD=0, CALL=1, RAISE=2, or -1 (no action / padding).
        """
        R, K = 4, 4
        matrix = [[-1] * K for _ in range(R)]
        # Group actions by round
        round_actions = defaultdict(list)
        for r, a in action_history:
            round_actions[r].append(a)
        for r in range(R):
            for i, a in enumerate(round_actions.get(r, [])):
                if i < K:
                    matrix[r][i] = float(a)
        return matrix

    # ==================================================================
    #  BNN inference
    # ==================================================================

    def _predict_proba(self, obs: Observation) -> tuple:
        """
        Return (mean_probs, uncertainty) from BNN for the given observation.
        Used internally by _encode_state for probability-distribution state.

        Returns:
            mean_probs: np.ndarray of shape (3,) — softmax probabilities
            uncertainty: float — std of predicted class probability
        """
        if not self.bnn_trained:
            return np.array([0.333, 0.333, 0.334]), 0.0
        features = self._encode_bnn_features(obs)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean_probs, uncertainty = self.bnn_model.predict_proba(x, mc_samples=self.mc_samples)
        return mean_probs[0], float(uncertainty[0])

    def _predict_opponent_label(self, obs: Observation) -> int:
        """
        Predict opponent hand strength using BNN with MC Dropout.

        Returns:
            0=weak, 1=mid, 2=strong (defaults to 1 if BNN is untrained)
        """
        if not self.bnn_trained:
            return 1  # mid

        features = self._encode_bnn_features(obs)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean_probs, uncertainty = self.bnn_model.predict_proba(x, mc_samples=self.mc_samples)
        pred = int(mean_probs[0].argmax())
        return pred

    def predict_with_uncertainty(self, obs: Observation):
        """
        Return (predicted_label, mean_probs, uncertainty) for diagnostics.
        """
        probs, uncertainty = self._predict_proba(obs)
        pred = int(probs.argmax())
        return pred, probs, uncertainty

    # ==================================================================
    #  MC Q-table learning
    # ==================================================================

    def learn_mc(self, trajectory: list[tuple], final_return: float) -> None:
        """
        Monte Carlo update for the Q-table.

        Walk the trajectory backward; for each (state, action) pair:
            Q(s, a) ← Q(s, a) + α * (G_t - Q(s, a))

        Args:
            trajectory: list of (state, action) tuples visited by this agent
            final_return: total return (reward) from this hand
        """
        for state, action in trajectory:
            td_error = final_return - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error

    def learn_sarsa(self, state: tuple, action: int, reward: float,
                    next_state: tuple, next_action: int, done: bool) -> None:
        """
        SARSA (TD) update for the Q-table.

        Q(s, a) ← Q(s, a) + α * [R + γ * Q(s', a') - Q(s, a)]

        Args:
            state       : encoded state before action
            action      : action taken (0/1/2)
            reward      : immediate reward (0 for intermediate, terminal reward at end)
            next_state  : next state after action (None if done)
            next_action : action in next_state (None if done)
            done        : whether episode is over
        """
        q_current = self.q_table[state][action]
        if done:
            td_target = reward
        else:
            q_next = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * q_next
        td_error = td_target - q_current
        self.q_table[state][action] += self.alpha * td_error

    # ==================================================================
    #  Epsilon management
    # ==================================================================

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ==================================================================
    #  Persistence
    # ==================================================================

    def save_model(self, filepath: str) -> None:
        """Save BNN weights + Q-table."""
        torch.save({
            "bnn_state_dict": self.bnn_model.state_dict(),
            "bnn_trained": self.bnn_trained,
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
        }, filepath)
        print(f"[NN_MCAgent] Model saved to {filepath} "
              f"(Q-table size={len(self.q_table)})")

    def load_model(self, filepath: str) -> None:
        """Load BNN weights + Q-table."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.bnn_model.load_state_dict(checkpoint["bnn_state_dict"])
        self.bnn_trained = checkpoint.get("bnn_trained", True)
        if "q_table" in checkpoint:
            self.q_table.clear()
            for state, q_vals in checkpoint["q_table"].items():
                self.q_table[state] = q_vals
        # Restore defaultdict behavior for unseen states
        self.q_table = type(self.q_table)(lambda: [0.0, 0.0, 0.0], self.q_table)
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        print(f"[NN_MCAgent] Model loaded from {filepath} "
              f"(Q-table size={len(self.q_table)}, bnn_trained={self.bnn_trained})")

    def get_q_table_size(self) -> int:
        return len(self.q_table)


# =========================================================================
#  BNN Training Data Utilities
# =========================================================================

def collect_bnn_training_data(env, num_hands: int = 5000,
                               mask_prob: float = 0.5,  # 50% mask: forces BNN to learn from public info
                               verbose: bool = True,
                               target_player: int = 1) -> tuple:
    """
    Collect labeled training data for the BNN.

    Feature masking strategy:
      - Compute opponent's TRUE hand features from env internal state.
      - With `mask_prob` probability, MASK opponent features (set to 0.5).
        The BNN must predict from public features alone.
      - With `1-mask_prob` probability, REVEAL opponent features.
        The BNN learns the direct hand→strength mapping.

    At inference time, opponent features are ALWAYS masked.

    Args:
        env: GameEngine with two agents
        num_hands: number of hands to collect
        mask_prob: probability of masking opponent features (default 0.5)
        verbose: print progress
        target_player: which player's hand strength to predict (0 or 1)

    Returns:
        X: np.ndarray of shape (N, 47) — BNN input features
        y: np.ndarray of shape (N,) — opponent strength labels {0,1,2}
        mask_flags: np.ndarray of shape (N,) — 1 if features were masked
    """
    from treys import Card

    X_list, y_list, mask_list = [], [], []
    dummy_agent = NN_MCAgent(name="DataCollector")
    dummy_agent._auto_record_self = False

    # observer_player: who observes (collects features)
    # target_player: whose hand strength to predict
    observer_player = 1 - target_player

    for hand in range(num_hands):
        dummy_agent.reset()
        obs = env.reset_hand()

        hand_features, hand_labels, hand_mask_flags = [], [], []

        done = False
        while not done:
            cp = env.current_player

            if cp == observer_player:
                # Compute target player's hand features from env internal state
                opp_hole = env.players[target_player].hole_cards
                opp_ranks = [Card.get_rank_int(c) for c in opp_hole]
                opp_rank_avg = sum(opp_ranks) / (len(opp_ranks) * 12.0)
                opp_suited = 1.0 if Card.get_suit_int(opp_hole[0]) == Card.get_suit_int(opp_hole[1]) else 0.0

                if len(obs.community_cards) >= 3:
                    opp_eq = compute_equity(opp_hole, obs.community_cards)
                else:
                    opp_eq = _preflop_opponent_equity(opp_hole)

                # mask_prob random mask: simulate inference-time "unknown opponent"
                is_masked = random.random() < mask_prob
                if not is_masked:
                    feat = dummy_agent._encode_bnn_features(
                        obs, opp_equity=opp_eq, opp_rank_avg=opp_rank_avg, opp_suited=opp_suited)
                else:
                    feat = dummy_agent._encode_bnn_features(obs)  # all masked

                hand_features.append(feat)
                hand_mask_flags.append(int(is_masked))
                label = _equity_to_strength_label(opp_eq)
                hand_labels.append(label)

                action = env.agents[cp].act(obs)
                dummy_agent.record_action(cp, action, obs.current_round)
            else:
                action = env.agents[cp].act(obs)
                dummy_agent.record_action(cp, action, obs.current_round)

            obs, reward, done, info = env.step(action)

        # Only use data from hands that went to showdown
        if info.get("result") and info["result"].winner is not None:
            X_list.extend(hand_features)
            y_list.extend(hand_labels)
            mask_list.extend(hand_mask_flags)

        if verbose and (hand + 1) % 1000 == 0:
            print(f"  Collected {len(X_list)} samples after {hand + 1} hands")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    mask_flags = np.array(mask_list, dtype=np.int64)
    return X, y, mask_flags


def _preflop_opponent_equity(hole_cards: list) -> float:
    """Rough preflop equity estimate based on card strength."""
    from treys import Card
    ranks = [Card.get_rank_int(c) for c in hole_cards]
    # Simple heuristic: average rank (higher is better)
    # ranks: 0=Two ... 12=Ace
    avg_rank = sum(ranks) / 2.0
    suited = Card.get_suit_int(hole_cards[0]) == Card.get_suit_int(hole_cards[1])
    equity = avg_rank / 12.0 * 0.4 + (0.2 if suited else 0.0) + 0.3
    paired = ranks[0] == ranks[1]
    if paired:
        equity += 0.1
    return min(max(equity, 0.05), 0.95)


def _equity_to_strength_label(equity: float) -> int:
    """Map equity to opponent strength label."""
    if equity > 0.6:
        return 2  # strong
    elif equity > 0.3:
        return 1  # mid
    else:
        return 0  # weak


# =========================================================================
#  BNN Training Loop
# =========================================================================

def train_bnn(model: BNNWithMCDropout, X: np.ndarray, y: np.ndarray,
              mask_flags: np.ndarray = None,
              epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
              val_split: float = 0.2,
              device: str = "cpu", verbose: bool = True) -> BNNWithMCDropout:
    """
    Train the BNN on labeled opponent-strength data with validation.

    Uses AdamW optimizer with weight decay, ReduceLROnPlateau scheduler,
    class-weighted loss, and gradient clipping.

    Args:
        model: BNNWithMCDropout instance
        X: features (N, 47)
        y: labels (N,) — {0,1,2}
        mask_flags: (N,) — 1 if opponent features were masked, else 0.
                    Used to report inference-time accuracy (masked-only ValAcc).
                    Default None means all samples considered.
        epochs: training epochs
        batch_size: mini-batch size
        lr: initial learning rate
        val_split: fraction of data used for validation
        device: 'cpu' or 'cuda'
        verbose: print progress

    Returns:
        trained model
    """
    # ---- Train/Val split ----
    n = len(X)
    n_val = int(n * val_split)
    indices = np.random.RandomState(42).permutation(n)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_t = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_t = torch.tensor(y[train_idx], dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if n_val > 0:
        X_v = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        y_v = torch.tensor(y[val_idx], dtype=torch.long).to(device)
        # Masked-only validation indices (simulates inference-time accuracy)
        if mask_flags is not None:
            val_mask_flags = mask_flags[val_idx]
            val_masked_idx = np.where(val_mask_flags == 1)[0]
            has_masked_val = len(val_masked_idx) > 0
            if has_masked_val:
                X_v_masked = X_v[val_masked_idx]
                y_v_masked = y_v[val_masked_idx]
        else:
            has_masked_val = False

    # ---- Class weights (inverse-frequency) ----
    y_train_np = y[train_idx]
    class_counts = np.bincount(y_train_np, minlength=3)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3  # normalize
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ---- AdamW optimizer with weight decay ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-6, verbose=False)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    if verbose:
        print(f"  Class weights: {class_weights}")
        print(f"  Optimizer: AdamW(lr={lr}, wd=1e-4), Scheduler: ReduceLROnPlateau")

    model.train()
    best_val_acc = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        train_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0

        # Validation accuracy
        val_acc = 0.0
        val_masked_acc = 0.0
        if n_val > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_v)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_v).float().mean().item()
                if has_masked_val:
                    vm_logits = model(X_v_masked)
                    vm_preds = vm_logits.argmax(dim=1)
                    val_masked_acc = (vm_preds == y_v_masked).float().mean().item()
            model.train()

        # Scheduler step based on validation accuracy
        if n_val > 0:
            scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if verbose and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1:>3}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"TrainAcc: {train_acc:.3f} | ValAcc: {val_acc:.3f} | "
                  f"Best: {best_val_acc:.3f} | LR: {current_lr:.2e}")

    # Final validation
    if n_val > 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_preds = val_logits.argmax(dim=1)
            final_val_acc = (val_preds == y_v).float().mean().item()
        model.train()
        if verbose:
            print(f"  Final ValAcc: {final_val_acc:.3f}  Best: {best_val_acc:.3f}")

    return model
