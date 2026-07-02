# agents/belief_net.py — MC-Dropout BNN + training utilities

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game.evaluator import (
    HAND_STRENGTH_SAMPLES,
    opponent_hand_strength,
    hand_strength_to_label,
    hand_strength_to_label_5class,
)
from agents.belief_features import BeliefFeatureEncoder

class BNNWithMCDropout(nn.Module):
    """MC-Dropout BNN for opponent strength (3- or 5-class)."""

    def __init__(self, input_dim=42, hidden_dims=(128, 64, 32),
                 num_classes=3, dropout_rate=0.15, multitask=False,
                 temperature: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.multitask = multitask
        self.temperature = temperature
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        self.shared_net = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate

        self.strength_head = nn.Linear(prev_dim, num_classes)

        if multitask:
            self.line_head = nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1),
            )
        else:
            self.line_head = None

    def forward(self, x):

        shared = self.shared_net(x)
        return self.strength_head(shared)

    def forward_multitask(self, x):
        shared = self.shared_net(x)
        strength_logits = self.strength_head(shared)
        line_logits = self.line_head(shared) if self.line_head is not None else None
        return strength_logits, line_logits

    def predict_proba(self, x, mc_samples=20, deterministic=False):
        if deterministic:
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits / self.temperature, dim=-1)
            mean_probs = probs.cpu().numpy()
            uncertainty = np.zeros(len(mean_probs))
            return mean_probs, uncertainty
        self.train()
        all_probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.forward(x)
                probs = F.softmax(logits / self.temperature, dim=-1)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.stack(all_probs, axis=0)
        mean_probs = all_probs.mean(axis=0)
        pred_classes = mean_probs.argmax(axis=1)
        batch_indices = np.arange(len(pred_classes))
        uncertainty = all_probs[:, batch_indices, pred_classes].std(axis=0)
        return mean_probs, uncertainty


def calibrate_temperature(model, X_val, y_val, device='cpu',
                          batch_size=128, T_range=(0.5, 3.0),
                          num_search=50, mc_samples=20):
    """Grid-search temperature T to minimize validation NLL."""
    import torch.nn.functional as F

    X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_val, dtype=torch.long).to(device)

    best_T = 1.0
    best_nll = float('inf')

    T_candidates = np.linspace(T_range[0], T_range[1], num_search)

    n_batches = int(np.ceil(len(X_val) / batch_size))

    for T in T_candidates:
        model.temperature = T
        total_nll = 0.0
        n_samples = 0

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(X_val))
            x_batch = X_t[start:end]
            y_batch = y_t[start:end]

            mean_probs, _ = model.predict_proba(x_batch, mc_samples=mc_samples)
            mean_probs_t = torch.tensor(mean_probs, dtype=torch.float32).to(device)
            log_probs = torch.log(mean_probs_t.clamp(min=1e-12))
            nll = F.nll_loss(log_probs, y_batch, reduction='sum')
            total_nll += nll.item()
            n_samples += len(y_batch)

        avg_nll = total_nll / n_samples
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_T = T

    model.temperature = best_T
    print(f"  Temperature calibration: optimal T={best_T:.3f} (NLL={best_nll:.4f})")
    return best_T


def collect_bnn_training_data(env, num_hands: int = 5000,
                               mask_prob: float = 0.5,
                               verbose: bool = True,
                               target_player: int = 1) -> tuple:
    """Labeled BNN samples from env trajectories (mask_prob masks opp slots)."""
    from treys import Card

    X_list, y_list, mask_list = [], [], []
    dummy_agent = BeliefFeatureEncoder()
    dummy_agent._auto_record_self = False

    observer_player = 1 - target_player

    for hand in range(num_hands):
        dummy_agent.reset()
        obs = env.reset_hand()

        hand_features, hand_labels, hand_mask_flags = [], [], []

        done = False
        while not done:
            cp = env.current_player

            if cp == observer_player:
                opp_hole = env.players[target_player].hole_cards
                opp_ranks = [Card.get_rank_int(c) for c in opp_hole]
                opp_rank_avg = sum(opp_ranks) / (len(opp_ranks) * 12.0)
                opp_suited = 1.0 if Card.get_suit_int(opp_hole[0]) == Card.get_suit_int(opp_hole[1]) else 0.0

                opp_strength = opponent_hand_strength(
                    opp_hole, obs.community_cards, num_samples=HAND_STRENGTH_SAMPLES)

                is_masked = random.random() < mask_prob
                if not is_masked:
                    feat = dummy_agent._encode_bnn_features(
                        obs, opp_equity=opp_strength, opp_rank_avg=opp_rank_avg,
                        opp_suited=opp_suited)
                else:
                    feat = dummy_agent._encode_bnn_features(obs)

                hand_features.append(feat)
                hand_mask_flags.append(int(is_masked))
                label = _equity_to_strength_label(opp_strength)
                hand_labels.append(label)

                action = env.agents[cp].act(obs)
                dummy_agent.record_action(cp, action, obs.current_round)
            else:
                action = env.agents[cp].act(obs)
                dummy_agent.record_action(cp, action, obs.current_round)

            obs, reward, done, info = env.step(action)

        X_list.extend(hand_features)
        y_list.extend(hand_labels)
        mask_list.extend(hand_mask_flags)

        if verbose and (hand + 1) % 1000 == 0:
            print(f"  Collected {len(X_list)} samples after {hand + 1} hands")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    mask_flags = np.array(mask_list, dtype=np.int64)
    return X, y, mask_flags



def _equity_to_strength_label(strength: float) -> int:
    return hand_strength_to_label(strength)


def _sarsa_action_to_opp_label(action: int) -> int:
    if action == 2:   # RAISE → ahead → opponent weak
        return 0
    elif action == 0: # FOLD  → behind → opponent strong
        return 2
    else:             # CALL  → marginal → opponent mid
        return 1


def _sarsa_q_to_opp_label_5class(action: int, q_vals: list, legal_actions: list) -> int:
    if action == 2:  # RAISE
        q_call = q_vals[1] if 1 in legal_actions else q_vals[0]
        margin = q_vals[2] - q_call
        return 0 if margin > 10 else 1  # VERY_WEAK vs WEAK
    elif action == 1:  # CALL
        q_fold = q_vals[0] if 0 in legal_actions else -999
        margin = q_vals[1] - q_fold
        return 2 if margin > 5 else 3  # MID vs STRONG
    else:  # FOLD
        return 4  # VERY_STRONG


def _equity_to_strength_label_5class(strength: float) -> int:
    return hand_strength_to_label_5class(strength)


def _opponent_strength_from_env(opp_hole: list, community_cards: list) -> float:
    return opponent_hand_strength(
        opp_hole, community_cards, num_samples=HAND_STRENGTH_SAMPLES)


#  SARSA Policy Distillation — Data Collection (NEW)

def collect_bnn_data_sarsa_distill(env, sarsa_agent,
                                    num_hands: int = 20000,
                                    mask_prob: float = 0.5,
                                    verbose: bool = True,
                                    observer_player: int = 0) -> tuple:
    """
    Collect BNN training data via SARSA policy distillation.

    KEY INSIGHT: Instead of requiring opponent's true hand strength
    (previously only collected at showdown), we use SARSA's
    greedy action at EVERY decision point as a proxy label:

        RAISE → opponent WEAK   |  CALL → opponent MID  |  FOLD → opponent STRONG

    This yields 5-10× more samples per hand. Note: the primary
    collect_bnn_training_data() now also records ALL decision points
    (not just showdown), eliminating the old selection bias.

    Feature masking (same as collect_bnn_training_data):
      - mask_prob of samples: opponent features set to 0.5 (inference sim)
      - 1-mask_prob of samples: reveal opponent's true hand for direct
        equity→action mapping signal

    Args:
        env: GameEngine with SARSA(ε=0) as observer and Expert as opponent
        sarsa_agent: trained SARSA agent (must have epsilon=0)
        num_hands: number of hands to collect
        mask_prob: masking probability (0.5 = half inference-style, half with hints)
        verbose: print progress
        observer_player: which player is SARSA (0 or 1)

    Returns:
        X: np.ndarray (N, 47) — BNN feature vectors
        y: np.ndarray (N,)    — opponent strength labels {0,1,2}
        mask_flags: np.ndarray (N,) — 1 if opponent features were masked
    """
    from treys import Card

    X_list, y_list, mask_list = [], [], []
    dummy = BeliefFeatureEncoder()
    dummy._auto_record_self = False

    target_player = 1 - observer_player

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

            if cp == observer_player:
                sarsa_state = sarsa_agent._encode_state(obs)
                q_vals = sarsa_agent.q_table[sarsa_state]
                legal = obs.legal_actions
                best_val = max(q_vals[a] for a in legal)
                best_actions = [a for a in legal if q_vals[a] == best_val]
                greedy_action = random.choice(best_actions)

                label = _sarsa_action_to_opp_label(greedy_action)

                is_masked = random.random() < mask_prob
                if not is_masked and len(env.players[target_player].hole_cards) == 2:
                    opp_hole = env.players[target_player].hole_cards
                    opp_ranks = [Card.get_rank_int(c) for c in opp_hole]
                    opp_rank_avg = sum(opp_ranks) / (len(opp_ranks) * 12.0)
                    opp_suited = 1.0 if Card.get_suit_int(opp_hole[0]) == Card.get_suit_int(opp_hole[1]) else 0.0
                    opp_strength = _opponent_strength_from_env(
                        opp_hole, obs.community_cards)
                    feat = dummy._encode_bnn_features(
                        obs, opp_equity=opp_strength, opp_rank_avg=opp_rank_avg,
                        opp_suited=opp_suited)
                else:
                    feat = dummy._encode_bnn_features(obs)  # fully masked

                X_list.append(feat)
                y_list.append(label)
                mask_list.append(int(is_masked))

                round_before = obs.current_round
                obs, reward, done, info = env.step(greedy_action)
                dummy.record_action(cp, greedy_action, round_before)
            else:
                round_before = obs.current_round
                opp_action = env.agents[cp].act(obs)
                obs, reward, done, info = env.step(opp_action)
                dummy.record_action(cp, opp_action, round_before)

        if verbose and (hand + 1) % 1000 == 0:
            print(f"  Distill: {len(X_list)} samples after {hand + 1} hands")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    mask_flags = np.array(mask_list, dtype=np.int64)
    return X, y, mask_flags


#  SARSA Policy Distillation — 5-Class Data Collection (NEW)

def collect_bnn_data_sarsa_distill_5class(env, sarsa_agent,
                                            num_hands: int = 20000,
                                            mask_prob: float = 0.5,
                                            verbose: bool = True,
                                            observer_player: int = 0) -> tuple:
    """
    Collect BNN training data via SARSA policy distillation — 5-class version.

    Uses Q-value margins for finer-grained opponent strength labels:
      RAISE, Q_raise - Q_call > 10  → VERY_WEAK   (0)
      RAISE, Q_raise - Q_call ≤ 10  → WEAK        (1)
      CALL,  Q_call - Q_fold > 5    → MID         (2)
      CALL,  Q_call - Q_fold ≤ 5    → STRONG      (3)
      FOLD                          → VERY_STRONG (4)

    Args:
        env: GameEngine with SARSA(ε=0) as observer and Expert as opponent
        sarsa_agent: trained SARSA agent
        num_hands: number of hands to collect
        mask_prob: masking probability
        verbose: print progress
        observer_player: which player is SARSA (0 or 1)

    Returns:
        X: np.ndarray (N, 47) — BNN feature vectors
        y: np.ndarray (N,)    — opponent strength labels {0,1,2,3,4}
        mask_flags: np.ndarray (N,) — 1 if opponent features were masked
    """
    from treys import Card

    X_list, y_list, mask_list = [], [], []
    dummy = BeliefFeatureEncoder()
    dummy._auto_record_self = False

    target_player = 1 - observer_player

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

            if cp == observer_player:
                sarsa_state = sarsa_agent._encode_state(obs)
                q_vals = sarsa_agent.q_table[sarsa_state]
                legal = obs.legal_actions
                best_val = max(q_vals[a] for a in legal)
                best_actions = [a for a in legal if q_vals[a] == best_val]
                greedy_action = random.choice(best_actions)

                label = _sarsa_q_to_opp_label_5class(greedy_action, q_vals, legal)

                is_masked = random.random() < mask_prob
                if not is_masked and len(env.players[target_player].hole_cards) == 2:
                    opp_hole = env.players[target_player].hole_cards
                    opp_ranks = [Card.get_rank_int(c) for c in opp_hole]
                    opp_rank_avg = sum(opp_ranks) / (len(opp_ranks) * 12.0)
                    opp_suited = 1.0 if Card.get_suit_int(opp_hole[0]) == Card.get_suit_int(opp_hole[1]) else 0.0
                    opp_strength = _opponent_strength_from_env(
                        opp_hole, obs.community_cards)
                    feat = dummy._encode_bnn_features(
                        obs, opp_equity=opp_strength, opp_rank_avg=opp_rank_avg,
                        opp_suited=opp_suited)
                else:
                    feat = dummy._encode_bnn_features(obs)  # fully masked

                X_list.append(feat)
                y_list.append(label)
                mask_list.append(int(is_masked))

                round_before = obs.current_round
                obs, reward, done, info = env.step(greedy_action)
                dummy.record_action(cp, greedy_action, round_before)
            else:
                round_before = obs.current_round
                opp_action = env.agents[cp].act(obs)
                obs, reward, done, info = env.step(opp_action)
                dummy.record_action(cp, opp_action, round_before)

        if verbose and (hand + 1) % 1000 == 0:
            print(f"  Distill(5-class): {len(X_list)} samples after {hand + 1} hands")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    mask_flags = np.array(mask_list, dtype=np.int64)
    return X, y, mask_flags


#  BNN Training Loop

def train_bnn(model: BNNWithMCDropout, X: np.ndarray, y: np.ndarray,
              mask_flags: np.ndarray = None,
              epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
              val_split: float = 0.2,
              device: str = "cpu", verbose: bool = True,
              use_label_smoothing: bool = False,
              use_cosine_schedule: bool = False) -> BNNWithMCDropout:
    """Train BNNWithMCDropout with optional masked samples."""
    num_classes = model.num_classes
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
        if mask_flags is not None:
            val_mask_flags = mask_flags[val_idx]
            val_masked_idx = np.where(val_mask_flags == 1)[0]
            has_masked_val = len(val_masked_idx) > 0
            if has_masked_val:
                X_v_masked = X_v[val_masked_idx]
                y_v_masked = y_v[val_masked_idx]
        else:
            has_masked_val = False

    y_train_np = y[train_idx]
    class_counts = np.bincount(y_train_np, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-6, verbose=False)
    smoothing = 0.1 if use_label_smoothing else 0.0
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=smoothing)

    if verbose:
        print(f"  Class weights: {class_weights}")
        sched_name = 'CosineAnnealing' if use_cosine_schedule else 'ReduceLROnPlateau'
        print(f"  Optimizer: AdamW(lr={lr}, wd=1e-4), Scheduler: {sched_name}")
        if use_label_smoothing:
            print(f"  Label smoothing: {smoothing}")

    model.train()
    best_val_acc = 0.0
    best_state = None
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

        if use_cosine_schedule:
            scheduler.step()
        elif n_val > 0:
            scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1:>3}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"TrainAcc: {train_acc:.3f} | ValAcc: {val_acc:.3f} | "
                  f"Best: {best_val_acc:.3f} | LR: {current_lr:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best model (ValAcc={best_val_acc:.3f})")

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


#  Line Consistency — auxiliary task for BNN multitask training

def _compute_line_consistency_label(aggression_score: float,
                                     equity: float) -> int:
    """
    Label whether opponent's betting line is consistent with hand strength.

    CONSISTENT (1): weak hand + passive, strong hand + aggressive
    INCONSISTENT (0): weak hand + aggressive (bluff), strong hand + passive

    Args:
        aggression_score: fraction of opponent actions that were raises [0,1]
        equity: opponent's true hand equity [0,1]

    Returns:
        1 = consistent, 0 = inconsistent
    """
    agg = aggression_score
    eq = equity
    if eq > 0.55 and agg > 0.3:
        return 1
    if eq < 0.3 and agg < 0.3:
        return 1
    if 0.3 <= eq <= 0.55 and 0.2 <= agg <= 0.5:
        return 1
    if eq < 0.3 and agg > 0.5:
        return 0
    if eq > 0.55 and agg < 0.15:
        return 0
    # Default: consistent
    return 1


def train_bnn_multitask(model: BNNWithMCDropout, X: np.ndarray,
                         y_strength: np.ndarray, y_line: np.ndarray,
                         epochs: int = 100, batch_size: int = 64,
                         lr: float = 1e-3, line_weight: float = 0.3,
                         val_split: float = 0.2,
                         device: str = "cpu", verbose: bool = True):
    """Multitask BNN training (strength + line head)."""
    num_classes = model.num_classes
    n = len(X)
    n_val = int(n * val_split)
    indices = np.random.RandomState(42).permutation(n)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_t = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_s_t = torch.tensor(y_strength[train_idx], dtype=torch.long).to(device)
    y_l_t = torch.tensor(y_line[train_idx], dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_t, y_s_t, y_l_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Class weights for strength head
    class_counts = np.bincount(y_strength[train_idx], minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    bce_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-6, verbose=False)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_ys, batch_yl in loader:
            optimizer.zero_grad()
            strength_logits, line_logits = model.forward_multitask(batch_x)

            ce_loss = ce_criterion(strength_logits, batch_ys)
            bce_loss = bce_criterion(line_logits.squeeze(-1), batch_yl)
            loss = ce_loss + line_weight * bce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_correct += (strength_logits.argmax(dim=1) == batch_ys).sum().item()
            total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validation
        val_acc = 0.0
        if n_val > 0:
            X_v = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
            y_v = torch.tensor(y_strength[val_idx], dtype=torch.long).to(device)
            model.eval()
            with torch.no_grad():
                val_logits = model(X_v)
                val_acc = (val_logits.argmax(dim=1) == y_v).float().mean().item()
            scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Multitask Epoch {epoch+1:>3}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"TrainAcc: {train_acc:.3f} | ValAcc: {val_acc:.3f} | "
                  f"Best: {best_val_acc:.3f}")

    if verbose:
        print(f"  Multitask Final ValAcc: {val_acc:.3f}  Best: {best_val_acc:.3f}")

    return model
