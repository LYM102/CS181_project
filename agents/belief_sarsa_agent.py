"""Tabular SARSA with BNN belief augmentation (shared base for L1/L2)."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from agents.belief_features import BeliefFeatureEncoder
from agents.belief_net import BNNWithMCDropout
from agents.belief_gating import (
    BeliefGatingNet,
    apply_learned_gating,
    load_gating_net,
    logits_from_q_values,
)
from game.engine import Observation
from game.evaluator import equity_to_bin, pot_to_bin


class BeliefSarsaAgent(BaseAgent, BeliefFeatureEncoder):
    """Tabular SARSA + BNN gated state (L1/L2 base)."""

    ACTION_SPACE = [0, 1, 2]
    OPP_STRENGTH_LABELS_3 = {"weak": 0, "mid": 1, "strong": 2}
    LABEL_TO_STR_3 = {0: "weak", 1: "mid", 2: "strong"}
    OPP_STRENGTH_LABELS_5 = {"very_weak": 0, "weak": 1, "mid": 2, "strong": 3, "very_strong": 4}
    LABEL_TO_STR_5 = {0: "very_weak", 1: "weak", 2: "mid", 3: "strong", 4: "very_strong"}
    BNN_FEATURE_DIM = 53
    VALID_STATE_MODES = ('prob3', 'compact', 'argmax', 'gated')

    def __init__(self, name: str = "BeliefSarsaAgent",
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.05,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 mc_samples: int = 20,
                 device: str = "cpu",
                 player_id: int = 0,
                 load_model_path: str = None,
                 state_mode: str = 'gated',
                 bnn_hidden_dims: tuple = (128, 64, 32),
                 bnn_dropout: float = 0.15,
                 bnn_confidence_threshold: float = 0.55,
                 gated_opp_coarse: str = '3class',
                 l0_backup_alpha: float = 0.0,
                 num_opp_classes: int = 3,
                 use_action_gating: bool = False,
                 gating_model_path: str = None,
                 gate_selective: bool = True,
                 gate_scale: float = 0.7):
        BaseAgent.__init__(self, name=name)
        BeliefFeatureEncoder.__init__(self)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma
        self.mc_samples = mc_samples
        self.device = device
        self.player_id = player_id
        self.state_mode = state_mode
        self.bnn_confidence_threshold = bnn_confidence_threshold
        assert gated_opp_coarse in ('3class', 'binary', 'exploit'), gated_opp_coarse
        self.gated_opp_coarse = gated_opp_coarse
        self.l0_backup_alpha = l0_backup_alpha
        self.num_opp_classes = num_opp_classes
        self.OPP_STRENGTH_LABELS = (self.OPP_STRENGTH_LABELS_5 if num_opp_classes == 5
                                    else self.OPP_STRENGTH_LABELS_3)
        self.LABEL_TO_STR = (self.LABEL_TO_STR_5 if num_opp_classes == 5
                             else self.LABEL_TO_STR_3)
        assert state_mode in self.VALID_STATE_MODES, f"Invalid state_mode: {state_mode}"
        assert num_opp_classes in (3, 5), f"num_opp_classes must be 3 or 5, got {num_opp_classes}"

        self.bnn_model = BNNWithMCDropout(
            input_dim=self.BNN_FEATURE_DIM,
            hidden_dims=bnn_hidden_dims,
            num_classes=num_opp_classes,
            dropout_rate=bnn_dropout,
        ).to(self.device)
        self.bnn_trained = False

        self.use_action_gating = use_action_gating
        self.gate_selective = gate_selective
        self.gate_scale = gate_scale
        self.gating_net: BeliefGatingNet | None = None
        self.gating_trained = False
        self.bluff_log: list = []

        self.q_table: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

        if load_model_path:
            self.load_model(load_model_path)
        if gating_model_path:
            self.load_gating_model(gating_model_path)

    def load_gating_model(self, filepath: str) -> None:
        self.gating_net = load_gating_net(filepath, device=self.device)
        self.gating_trained = True
        self.use_action_gating = True
        print(f"[BeliefSarsaAgent] Gating model loaded from {filepath}")

    def _select_gated_action(self, obs: Observation, state: tuple,
                             legal_actions: list[int]) -> int:
        q_vals = self.q_table[state]
        base_logits = logits_from_q_values(np.array(q_vals, dtype=np.float32))
        if self.bnn_trained:
            belief_probs, uncertainty = self._predict_proba(obs)
        else:
            belief_probs = np.ones(3, dtype=np.float32) / 3.0
            uncertainty = 1.0
        probs = apply_learned_gating(
            self.gating_net, base_logits, belief_probs, uncertainty,
            legal_actions, obs, self._opp_actions, device=self.device,
            log=self.bluff_log, gate_scale=self.gate_scale,
            selective=self.gate_selective)
        masked = np.array([probs[a] if a in legal_actions else -1.0 for a in range(3)])
        return int(masked.argmax())

    def act(self, obs: Observation) -> int:
        cc_count = len(obs.community_cards)
        if cc_count < self._prev_community_count:
            self.reset()
        self._prev_community_count = cc_count

        state = self._encode_state(obs)
        legal_actions = obs.legal_actions

        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        elif (self.use_action_gating and self.gating_trained and self.gating_net is not None):
            action = self._select_gated_action(obs, state, legal_actions)
        else:
            q_vals = self.q_table[state]
            best_value = max(q_vals[a] for a in legal_actions)
            best_actions = [a for a in legal_actions if q_vals[a] == best_value]
            action = random.choice(best_actions)

        if self._auto_record_self:
            self._self_actions.append((obs.current_round, action))
        return action

    def reset(self) -> None:
        BeliefFeatureEncoder.reset(self)

    def update(self, obs: Observation, action: int, reward: float,
               next_obs: Observation, done: bool) -> None:
        pass

    def _gated_opp_bucket(self, opp_class: int, max_prob: float):
        if max_prob < self.bnn_confidence_threshold:
            return None
        if self.gated_opp_coarse == 'binary':
            return 0 if opp_class == 0 else 1
        if self.gated_opp_coarse == 'exploit':
            if opp_class == 1:
                return None
            return 0 if opp_class == 0 else 1
        return opp_class

    def _l0_base_key(self, state: tuple):
        if len(state) == 5:
            return state
        if len(state) == 6:
            return state[:5]
        return None

    def _belief_bucket_keys(self, base: tuple) -> list[tuple]:
        if self.gated_opp_coarse in ('binary', 'exploit'):
            return [base + (0,), base + (1,)]
        return [base + (c,) for c in range(self.num_opp_classes)]

    def _encode_state(self, obs: Observation) -> tuple:
        h_code = equity_to_bin(obs.equity, bins=20)
        pot_bin_val = pot_to_bin(obs.pot)
        p_code = len(obs.community_cards)

        if self.bnn_trained:
            probs, _ = self._predict_proba(obs)

            if self.state_mode == 'prob3':
                o_weak = min(int(probs[0] * 3), 2)
                o_mid = min(int(probs[1] * 3), 2)
                o_strong = min(int(probs[2] * 3), 2)
                return (h_code, p_code, obs.betting_level, pot_bin_val,
                        obs.position, o_weak, o_mid, o_strong)

            if self.state_mode == 'compact':
                opp_class = int(probs.argmax())
                confidence = 1 if float(probs.max()) >= 0.5 else 0
                return (h_code, p_code, obs.betting_level, pot_bin_val,
                        obs.position, opp_class, confidence)

            if self.state_mode == 'gated':
                opp_class = int(probs.argmax())
                max_prob = float(probs.max())
                base = (h_code, p_code, obs.betting_level, pot_bin_val, obs.position)
                bucket = self._gated_opp_bucket(opp_class, max_prob)
                if bucket is not None:
                    return base + (bucket,)
                return base

            opp_class = int(probs.argmax())
            return (h_code, p_code, obs.betting_level, pot_bin_val, obs.position, opp_class)

        mid_class = self.num_opp_classes // 2
        if self.state_mode == 'prob3':
            t = ((mid_class,) * 3 if self.num_opp_classes == 3
                 else (0, 0, 1, 0, 0))
            return (h_code, p_code, obs.betting_level, pot_bin_val, obs.position, *t)
        if self.state_mode == 'compact':
            return (h_code, p_code, obs.betting_level, pot_bin_val,
                    obs.position, mid_class, 0)
        if self.state_mode == 'gated':
            return (h_code, p_code, obs.betting_level, pot_bin_val, obs.position)
        return (h_code, p_code, obs.betting_level, pot_bin_val, obs.position, mid_class)

    def _predict_proba(self, obs: Observation) -> tuple:
        if not self.bnn_trained:
            uniform = np.ones(self.num_opp_classes) / self.num_opp_classes
            return uniform, 0.0
        features = self._encode_bnn_features(obs)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean_probs, uncertainty = self.bnn_model.predict_proba(x, mc_samples=self.mc_samples)
        return mean_probs[0], float(uncertainty[0])

    def _predict_opponent_label(self, obs: Observation) -> int:
        if not self.bnn_trained:
            return self.num_opp_classes // 2
        probs, _ = self._predict_proba(obs)
        return int(probs.argmax())

    def predict_with_uncertainty(self, obs: Observation):
        probs, uncertainty = self._predict_proba(obs)
        return int(probs.argmax()), probs, uncertainty

    def learn_mc(self, trajectory: list[tuple], final_return: float) -> None:
        for state, action in trajectory:
            td_error = final_return - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error

    def learn_sarsa(self, state: tuple, action: int, reward: float,
                    next_state: tuple, next_action: int, done: bool) -> None:
        q_current = self.q_table[state][action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_error = td_target - q_current
        self.q_table[state][action] += self.alpha * td_error
        if self.l0_backup_alpha > 0 and self.state_mode == 'gated':
            base = self._l0_base_key(state)
            if base is not None and base != state:
                self.q_table[base][action] += self.l0_backup_alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str) -> None:
        torch.save({
            "bnn_state_dict": self.bnn_model.state_dict(),
            "bnn_trained": self.bnn_trained,
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "bnn_confidence_threshold": self.bnn_confidence_threshold,
            "gated_opp_coarse": self.gated_opp_coarse,
            "l0_backup_alpha": self.l0_backup_alpha,
            "state_mode": self.state_mode,
        }, filepath)
        print(f"[BeliefSarsaAgent] Model saved to {filepath} (Q-table size={len(self.q_table)})")

    def load_model(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        bnn_state = checkpoint["bnn_state_dict"]

        layers = []
        for key in sorted(bnn_state.keys()):
            if key.startswith('shared_net.') and key.endswith('.weight'):
                layers.append(bnn_state[key].shape)
        if layers:
            hidden_dims = tuple(s[0] for s in layers)
            num_classes = bnn_state.get('strength_head.weight', torch.zeros(3, 1)).shape[0]
            current_dims = tuple(l.out_features for l in self.bnn_model.shared_net
                                 if isinstance(l, nn.Linear))
            if hidden_dims != current_dims:
                print(f"[BeliefSarsaAgent] Reconstructing BNN: {current_dims} → {hidden_dims}")
                self.bnn_model = BNNWithMCDropout(
                    input_dim=self.BNN_FEATURE_DIM,
                    hidden_dims=hidden_dims,
                    num_classes=num_classes,
                    dropout_rate=0.15,
                ).to(self.device)

        self.bnn_model.load_state_dict(bnn_state)
        self.bnn_trained = checkpoint.get("bnn_trained", True)
        if "q_table" in checkpoint:
            self.q_table.clear()
            for state, q_vals in checkpoint["q_table"].items():
                self.q_table[state] = q_vals
        self.q_table = type(self.q_table)(lambda: [0.0, 0.0, 0.0], self.q_table)
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "bnn_confidence_threshold" in checkpoint:
            self.bnn_confidence_threshold = checkpoint["bnn_confidence_threshold"]
        if "gated_opp_coarse" in checkpoint:
            self.gated_opp_coarse = checkpoint["gated_opp_coarse"]
        if "l0_backup_alpha" in checkpoint:
            self.l0_backup_alpha = checkpoint["l0_backup_alpha"]
        if "state_mode" in checkpoint:
            self.state_mode = checkpoint["state_mode"]
        print(f"[BeliefSarsaAgent] Model loaded from {filepath} "
              f"(Q-table size={len(self.q_table)}, bnn_trained={self.bnn_trained}, "
              f"gated={self.gated_opp_coarse}, τ={self.bnn_confidence_threshold})")

    def get_q_table_size(self) -> int:
        return len(self.q_table)

    def warm_start_from_l0(self, sarsa_qtable_path: str,
                           replicate_belief: bool = True) -> int:
        import pickle
        with open(sarsa_qtable_path, "rb") as f:
            l0_table = pickle.load(f)

        copied = 0
        for state, q_vals in l0_table.items():
            if len(state) != 5:
                continue
            self.q_table[state] = list(q_vals)
            copied += 1
            if replicate_belief and self.state_mode == "gated":
                for key in self._belief_bucket_keys(state):
                    self.q_table[key] = list(q_vals)

        print(f"[BeliefSarsaAgent] Warm-started {copied} L0 states from {sarsa_qtable_path} "
              f"(Q-table size={len(self.q_table)})")
        return copied

    def q_table_coverage_stats(self) -> dict:
        l0_keys, belief_keys = 0, 0
        for key in self.q_table:
            if len(key) == 5:
                l0_keys += 1
            elif len(key) == 6:
                belief_keys += 1
        l0_theoretical = 20 * 4 * 6 * 7 * 2
        split_factor = 3 if self.gated_opp_coarse == '3class' else 2
        belief_theoretical = l0_theoretical * split_factor
        return {
            "total": len(self.q_table),
            "l0_keys": l0_keys,
            "belief_keys": belief_keys,
            "l0_coverage_pct": 100.0 * l0_keys / l0_theoretical,
            "belief_coverage_pct": 100.0 * belief_keys / belief_theoretical,
            "gated_opp_coarse": self.gated_opp_coarse,
            "confidence_threshold": self.bnn_confidence_threshold,
        }
