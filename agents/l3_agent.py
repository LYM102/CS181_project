"""L3 agent: CFR-distilled neural policy with BNN belief and residual gate."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from agents.belief_features import BeliefFeatureEncoder
from agents.belief_net import BNNWithMCDropout
from agents.belief_gating import (
    BeliefGatingNet,
    apply_learned_gating,
    load_gating_net,
    opp_aggression_score,
    line_inconsistency_score,
)
from game.evaluator import compute_hand_strength, HAND_STRENGTH_SAMPLES

class ResidualBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out = self.act(self.ln1(self.linear1(x)))
        out = self.dropout(out)
        out = self.ln2(self.linear2(out))
        return self.act(out + residual)


class BNN_PolicyNet(nn.Module):
    """53-dim features → action logits."""

    def __init__(self, input_dim=53, hidden_dims=(256, 128, 64),
                 dropout_rate=0.15, use_residual=True, use_layernorm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        if use_residual:
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(ResidualBlock(prev_dim, h_dim, dropout_rate))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 3))
            self.net = nn.Sequential(*layers)
        else:
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                if use_layernorm:
                    layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 3))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_action(self, x, mc_samples=20):
        self.train()
        all_probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.stack(all_probs, axis=0)
        mean_probs = all_probs.mean(axis=0)
        greedy_actions = mean_probs.argmax(axis=1)
        batch_idx = np.arange(len(greedy_actions))
        uncertainty = all_probs[:, batch_idx, greedy_actions].std(axis=0)
        return mean_probs, uncertainty

    def get_arch_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "use_residual": self.use_residual,
            "use_layernorm": self.use_layernorm,
        }
class L3Agent(BaseAgent):
    """Neural policy + BNN belief + learned gate (L3)."""

    ACTION_SPACE = [0, 1, 2]
    FEATURE_DIM = 53
    BELIEF_EQUITY_CENTROIDS = (0.2, 0.5, 0.8)

    def __init__(self, name: str = "L3Agent",
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.05,
                 mc_samples: int = 20,
                 device: str = "cpu",
                 player_id: int = 0,
                 load_model_path: str = None,
                 belief_model_path: str = None,
                 use_belief: bool = True,
                 use_learned_gating: bool = True,
                 gating_model_path: str = None,
                 hidden_dims: tuple = (256, 128, 64),
                 dropout_rate: float = 0.15,
                 use_residual: bool = True,
                 use_layernorm: bool = True,
                 gate_selective: bool = False,
                 gate_scale: float = 1.0,
                 deterministic_belief: bool = False):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.mc_samples = mc_samples
        self.device = device
        self.player_id = player_id
        self._hidden_dims = hidden_dims
        self._dropout_rate = dropout_rate
        self._use_residual = use_residual
        self._use_layernorm = use_layernorm
        self.use_belief = use_belief
        self.use_learned_gating = use_learned_gating
        self.gate_selective = gate_selective
        self.gate_scale = gate_scale
        self.deterministic_belief = deterministic_belief
        self.gating_net: BeliefGatingNet | None = None
        self.gating_trained = False
        self.bluff_log: list = []

        self.policy_net = BNN_PolicyNet(
            input_dim=self.FEATURE_DIM,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_layernorm=use_layernorm,
        ).to(self.device)

        self.belief_net = BNNWithMCDropout(
            input_dim=BeliefFeatureEncoder.BNN_FEATURE_DIM,
            hidden_dims=(128, 64, 32),
            num_classes=3,
            dropout_rate=0.15,
        ).to(self.device)
        self.belief_trained = False

        self._feat_builder = BeliefFeatureEncoder()
        self._feat_builder._auto_record_self = False
        self._prev_community_count = 0

        self._ewc_fisher = {}
        self._ewc_params = {}
        self._ewc_lambda = 0.0

        if load_model_path:
            self.load_model(load_model_path)
        if belief_model_path:
            self.load_belief_model(belief_model_path)
        if gating_model_path:
            self.load_gating_model(gating_model_path)

    def load_gating_model(self, filepath: str) -> None:
        self.gating_net = load_gating_net(filepath, device=self.device)
        self.gating_trained = True
        self.use_learned_gating = True
        print(f"[L3Agent] Gating model loaded from {filepath}")

    def _predict_opponent_belief(self, obs: Observation):
        """Run BNN belief net on masked public features."""
        if not self.belief_trained:
            uniform = np.ones(3, dtype=np.float32) / 3.0
            return uniform, 1.0
        bnn_input = self._feat_builder._encode_bnn_features(obs)
        x = torch.tensor(bnn_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean_probs, uncertainty = self.belief_net.predict_proba(
            x, mc_samples=self.mc_samples,
            deterministic=getattr(self, 'deterministic_belief', False))
        return mean_probs[0], float(uncertainty[0])

    def _belief_to_opp_features(self, belief_probs: np.ndarray,
                                 uncertainty: float):
        """Map BNN belief distribution to policy input slots [44:47]."""
        centroids = self.BELIEF_EQUITY_CENTROIDS
        opp_equity = float(sum(belief_probs[i] * centroids[i] for i in range(3)))
        # Signed strength: P(strong) − P(weak), mapped to [0, 1]
        opp_rank_avg = float((belief_probs[2] - belief_probs[0] + 1.0) / 2.0)
        # Confidence from MC Dropout uncertainty (lower std → higher confidence)
        confidence = max(0.0, 1.0 - min(uncertainty / 0.25, 1.0))
        return opp_equity, opp_rank_avg, confidence

    def _build_policy_features(self, obs: Observation):
        if self.use_belief and self.belief_trained:
            belief_probs, uncertainty = self._predict_opponent_belief(obs)
            opp_eq, opp_rank, confidence = self._belief_to_opp_features(
                belief_probs, uncertainty)
            features = self._feat_builder._encode_bnn_features(
                obs, opp_equity=opp_eq, opp_rank_avg=opp_rank, opp_suited=confidence)
        else:
            belief_probs = np.ones(3, dtype=np.float32) / 3.0
            uncertainty = 1.0
            features = self._feat_builder._encode_bnn_features(obs)


        return features, belief_probs, uncertainty

    def _opp_aggression_score(self) -> float:
        """Fraction of opponent actions that were raises (0..1)."""
        return opp_aggression_score(self._feat_builder._opp_actions)

    def _opp_last_street_raise(self, obs: Observation) -> bool:
        """True if opponent raised on the current betting street."""
        r = obs.current_round
        return any(a == 2 for round_num, a in self._feat_builder._opp_actions
                   if round_num == r)

    def encode_policy_features(self, obs: Observation) -> np.ndarray:
        features, _, _ = self._build_policy_features(obs)
        return features

    def _apply_belief_gating(self, action_logits: np.ndarray,
                              belief_probs: np.ndarray,
                              uncertainty: float,
                              legal: list[int],
                              obs: Observation = None) -> np.ndarray:
        if not (self.use_belief and self.belief_trained):
            masked = np.array([action_logits[a] if a in legal else -1e9 for a in range(3)])
            ex = np.exp(masked - masked.max())
            return (ex / ex.sum()).astype(np.float32)

        if not (self.use_learned_gating and self.gating_trained
                and self.gating_net is not None):
            masked = np.array([action_logits[a] if a in legal else -1e9 for a in range(3)])
            ex = np.exp(masked - masked.max())
            return (ex / ex.sum()).astype(np.float32)

        return apply_learned_gating(
            self.gating_net,
            action_logits.astype(np.float32),
            belief_probs,
            uncertainty,
            legal,
            obs,
            self._feat_builder._opp_actions,
            device=self.device,
            log=self.bluff_log if hasattr(self, "bluff_log") else None,
            gate_scale=self.gate_scale,
            selective=self.gate_selective,
        )

    def _select_action(self, obs: Observation) -> int:
        features, belief_probs, uncertainty = self._build_policy_features(obs)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.policy_net.eval()
        with torch.no_grad():
            logits = self.policy_net(x).squeeze(0).cpu().numpy()

        legal = obs.legal_actions
        probs = self._apply_belief_gating(logits, belief_probs, uncertainty, legal, obs)
        masked_probs = np.array([probs[a] if a in legal else -1.0 for a in range(3)])
        return int(masked_probs.argmax())

    def act(self, obs: Observation) -> int:
        cc_count = len(obs.community_cards)
        if cc_count < self._prev_community_count:
            self.reset()
        self._prev_community_count = cc_count

        legal = obs.legal_actions
        if random.random() < self.epsilon:
            return random.choice(legal)
        return self._select_action(obs)

    def reset(self) -> None:
        self._feat_builder.reset()
        self._prev_community_count = 0

    def update(self, obs, action, reward, next_obs, done):
        pass

    def record_action(self, player: int, action: int, round_num: int) -> None:
        self._feat_builder.record_action(player, action, round_num)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def init_dagger(self, lr: float = 1e-4, capacity: int = 50000):
        self.dagger_buffer: list[tuple] = []
        self.dagger_capacity = capacity
        self.dagger_lr = lr
        self.dagger_optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, weight_decay=1e-4)

    def add_dagger_sample(self, features: np.ndarray, sarsa_action: int) -> None:
        self.dagger_buffer.append((features.copy(), sarsa_action))
        if len(self.dagger_buffer) > self.dagger_capacity:
            self.dagger_buffer.pop(0)

    def reset_dagger(self) -> None:
        self.dagger_buffer = []

    def train_dagger(self, epochs: int = 10, batch_size: int = 128,
                     ewc_lambda: float = 0.0):
        if len(self.dagger_buffer) < batch_size:
            return 0.0, 0.0

        X_d = np.stack([s[0] for s in self.dagger_buffer])
        y_d = np.array([s[1] for s in self.dagger_buffer], dtype=np.int64)

        X_t = torch.tensor(X_d, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_d, dtype=torch.long).to(self.device)

        class_counts = np.bincount(y_d, minlength=3)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * 3
        cw_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=cw_t)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        self.policy_net.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                self.dagger_optimizer.zero_grad()
                logits = self.policy_net(batch_x)
                loss = criterion(logits, batch_y)

                if ewc_lambda > 0 and self._ewc_fisher:
                    ewc_loss = self._compute_ewc_loss()
                    loss = loss + ewc_lambda * ewc_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
                self.dagger_optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
                total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        acc = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, acc

    def init_ewc(self, dataloader, ewc_lambda: float = 100.0, n_samples: int = 500):
        self._ewc_lambda = ewc_lambda
        self._ewc_fisher = {}
        self._ewc_params = {}

        for name, param in self.policy_net.named_parameters():
            self._ewc_params[name] = param.data.clone()

        self.policy_net.train()
        criterion = nn.CrossEntropyLoss()
        sample_count = 0

        for batch_x, batch_y in dataloader:
            if sample_count >= n_samples:
                break
            self.policy_net.zero_grad()
            logits = self.policy_net(batch_x.to(self.device))
            loss = criterion(logits, batch_y.to(self.device))
            loss.backward()

            for name, param in self.policy_net.named_parameters():
                if param.grad is not None:
                    if name not in self._ewc_fisher:
                        self._ewc_fisher[name] = param.grad.data.clone() ** 2
                    else:
                        self._ewc_fisher[name] += param.grad.data.clone() ** 2
            sample_count += batch_x.size(0)

        for name in self._ewc_fisher:
            self._ewc_fisher[name] /= max(sample_count, 1)

        print(f"[EWC] Fisher computed on {sample_count} samples, "
              f"lambda={ewc_lambda}")

    def _compute_ewc_loss(self) -> torch.Tensor:
        ewc_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.policy_net.named_parameters():
            if name in self._ewc_fisher and name in self._ewc_params:
                fisher = self._ewc_fisher[name].to(self.device)
                frozen = self._ewc_params[name].to(self.device)
                ewc_loss += (fisher * (param - frozen) ** 2).sum()
        return ewc_loss

    def train_rl_step(self, ewc_lambda: float = 0.0):
        if len(self.rl_trajectory) == 0:
            return 0.0, 0

        total_return = sum(s["reward"] for s in self.rl_trajectory)
        self.rl_baseline = (self.rl_baseline_alpha * total_return +
                            (1 - self.rl_baseline_alpha) * self.rl_baseline)
        advantage = total_return - self.rl_baseline

        total_loss = 0.0
        n_steps = len(self.rl_trajectory)

        self.policy_net.train()
        for step_record in self.rl_trajectory:
            x = torch.tensor(step_record["features"], dtype=torch.float32).unsqueeze(0).to(self.device)
            a = torch.tensor([step_record["action"]], dtype=torch.long).to(self.device)

            logits = self.policy_net(x)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_prob = log_probs[0, a[0]]

            pg_loss = -selected_log_prob * advantage

            if ewc_lambda > 0 and self._ewc_fisher:
                ewc_loss = self._compute_ewc_loss()
                pg_loss = pg_loss + ewc_lambda * ewc_loss

            self.rl_optimizer.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.rl_optimizer.step()
            total_loss += pg_loss.item()

        self.rl_trajectory = []
        avg_loss = total_loss / n_steps
        return avg_loss, n_steps

    def save_model(self, filepath: str) -> None:
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        payload = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "arch_config": self.policy_net.get_arch_config(),
            "epsilon": self.epsilon,
        }
        if self.belief_trained:
            payload["belief_net_state_dict"] = self.belief_net.state_dict()
            payload["belief_trained"] = True
        torch.save(payload, filepath)
        print(f"[L3Agent] Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        if "arch_config" in checkpoint:
            arch = checkpoint["arch_config"]
            if (arch.get("hidden_dims") != self._hidden_dims or
                    arch.get("use_residual", False) != self._use_residual):
                self.policy_net = BNN_PolicyNet(
                    input_dim=arch.get("input_dim", self.FEATURE_DIM),
                    hidden_dims=arch.get("hidden_dims", self._hidden_dims),
                    dropout_rate=arch.get("dropout_rate", self._dropout_rate),
                    use_residual=arch.get("use_residual", False),
                    use_layernorm=arch.get("use_layernorm", False),
                ).to(self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        if checkpoint.get("belief_trained") and "belief_net_state_dict" in checkpoint:
            self.belief_net.load_state_dict(checkpoint["belief_net_state_dict"])
            self.belief_trained = True
        print(f"[L3Agent] Model loaded from {filepath}")

    def load_belief_model(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        state_dict = checkpoint.get("belief_net_state_dict") or checkpoint.get("bnn_state_dict")
        if state_dict is None:
            raise KeyError(
                f"No belief weights in {filepath} "
                "(expected 'belief_net_state_dict' or 'bnn_state_dict')")
        
        layers = []
        for key in sorted(state_dict.keys()):
            if key.startswith('shared_net.') and key.endswith('.weight'):
                layers.append(state_dict[key].shape)
        if layers:
            hidden_dims = tuple(s[0] for s in layers)  # output dims of each layer
            num_classes = state_dict.get('strength_head.weight', 
                                         state_dict.get('belief_net.strength_head.weight',
                                         torch.zeros(3,1))).shape[0]
            
            current_dims = tuple(l.out_features for l in self.belief_net.shared_net 
                                 if isinstance(l, nn.Linear))
            if hidden_dims != current_dims:
                print(f"[L3Agent] Reconstructing belief net: {current_dims} → {hidden_dims}")
                self.belief_net = BNNWithMCDropout(
                    input_dim=BeliefFeatureEncoder.BNN_FEATURE_DIM,
                    hidden_dims=hidden_dims,
                    num_classes=num_classes,
                    dropout_rate=0.1,
                ).to(self.device)
        
        self.belief_net.load_state_dict(state_dict)
        self.belief_trained = checkpoint.get("belief_trained",
                                               checkpoint.get("bnn_trained", True))
        print(f"[L3Agent] Belief net loaded from {filepath}")

    def save_belief_model(self, filepath: str) -> None:
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save({
            "belief_net_state_dict": self.belief_net.state_dict(),
            "belief_trained": self.belief_trained,
        }, filepath)
        print(f"[L3Agent] Belief net saved to {filepath}")


def collect_expert_policy_data(expert_agent, num_hands: int = 30000,
                                mask_prob: float = 0.5,
                                verbose: bool = True) -> tuple:
    """Expert soft-label trajectories for KL distillation."""
    import random as _random
    from treys import Card
    from agents.random_agent import RandomAgent
    from game.engine import GameEngine

    X_list, y_prob_list, mask_list = [], [], []
    dummy = BeliefFeatureEncoder()
    dummy._auto_record_self = False

    opponent = RandomAgent(name="Random_Opp")
    env = GameEngine(expert_agent, opponent)

    for hand in range(num_hands):
        dummy.reset()
        obs = env.reset_hand()
        done = False
        step_count = 0

        while not done:
            step_count += 1
            if step_count > 50:
                break

            cp = env.current_player

            if cp == 0:  # Expert's turn
                # Get Expert's mixed strategy as soft label
                expert_probs = expert_agent.get_action_probs(obs)

                # Build features with optional opponent masking
                is_masked = _random.random() < mask_prob
                if not is_masked and len(env.players[1].hole_cards) == 2:
                    opp_hole = env.players[1].hole_cards
                    opp_ranks = [Card.get_rank_int(c) for c in opp_hole]
                    opp_rank_avg = sum(opp_ranks) / (len(opp_ranks) * 12.0)
                    opp_suited = 1.0 if Card.get_suit_int(opp_hole[0]) == Card.get_suit_int(opp_hole[1]) else 0.0
                    opp_strength = compute_hand_strength(
                        opp_hole, obs.community_cards, num_samples=HAND_STRENGTH_SAMPLES)
                    feat = dummy._encode_bnn_features(
                        obs, opp_equity=opp_strength, opp_rank_avg=opp_rank_avg,
                        opp_suited=opp_suited)
                else:
                    feat = dummy._encode_bnn_features(obs)

                X_list.append(feat)
                y_prob_list.append(np.array(expert_probs, dtype=np.float32))
                mask_list.append(int(is_masked))

                # Expert acts (sample from probs)
                action = expert_agent.act(obs)
                round_before = obs.current_round
                obs, reward, done, info = env.step(action)
                dummy.record_action(cp, action, round_before)
            else:  # Opponent's turn
                round_before = obs.current_round
                opp_action = env.agents[cp].act(obs)
                obs, reward, done, info = env.step(opp_action)
                dummy.record_action(cp, opp_action, round_before)

        if verbose and (hand + 1) % 1000 == 0:
            print(f"  ExpertDistill: {len(X_list)} samples after {hand + 1} hands")

    X = np.array(X_list, dtype=np.float32)
    y_probs = np.array(y_prob_list, dtype=np.float32)
    mask_flags = np.array(mask_list, dtype=np.int64)

    if verbose:
        print(f"\n  Collected {len(X)} samples in total")
        for i, name in enumerate(["FOLD", "CALL", "RAISE"]):
            avg_prob = y_probs[:, i].mean()
            print(f"    {name}: avg_prob={avg_prob:.3f}")
    return X, y_probs, mask_flags


def train_bnn_policy_kl(model: BNN_PolicyNet, X: np.ndarray,
                         y_probs: np.ndarray,
                         mask_flags: np.ndarray = None,
                         epochs: int = 150, batch_size: int = 64,
                         lr: float = 5e-4, alpha: float = 0.3,
                         temperature: float = 3.0,
                         val_split: float = 0.15,
                         device: str = "cpu", verbose: bool = True):
    """KL distillation from Expert mixed strategies."""
    # Convert soft probs to hard labels for CE
    y_hard = y_probs.argmax(axis=1).astype(np.int64)

    n = len(X)
    n_val = int(n * val_split)
    indices = np.random.RandomState(42).permutation(n)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train_hard = torch.tensor(y_hard[train_idx], dtype=torch.long).to(device)
    y_train_soft = torch.tensor(y_probs[train_idx], dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_train, y_train_hard, y_train_soft)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X_val, y_val = None, None
    if n_val > 0:
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        y_val = torch.tensor(y_hard[val_idx], dtype=torch.long).to(device)

    # Class-balanced weights for hard CE
    class_counts = np.bincount(y_hard[train_idx], minlength=3)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6, verbose=False)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for batch_x, batch_y_hard, batch_y_soft in loader:
            optimizer.zero_grad()
            student_logits = model(batch_x)

            # Hard label CE loss
            ce_loss = ce_criterion(student_logits, batch_y_hard)

            # Soft distillation loss (KL divergence)
            student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_soft = (batch_y_soft + 1e-8) / (batch_y_soft + 1e-8).sum(dim=-1, keepdim=True)
            kl_loss = kl_criterion(student_log_soft, teacher_soft) * (temperature ** 2)

            loss = alpha * ce_loss + (1 - alpha) * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_correct += (student_logits.argmax(dim=1) == batch_y_hard).sum().item()

        avg_loss = total_loss / len(train_idx)
        train_acc = total_correct / len(train_idx)

        val_loss = float('inf')
        val_acc = 0.0
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = ce_criterion(val_logits, y_val).item()
                val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
            model.train()
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if verbose and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  DistillKL Epoch {epoch + 1:>3}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"TrainAcc: {train_acc:.3f} | ValLoss: {val_loss:.4f} | "
                  f"ValAcc: {val_acc:.3f} | BestValAcc: {best_val_acc:.3f} | LR: {current_lr:.2e}")

    if verbose:
        print(f"  DistillKL Final ValAcc: {val_acc:.3f}  Best: {best_val_acc:.3f}")

    return model
