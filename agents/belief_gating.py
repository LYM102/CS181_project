"""Residual gating network g_θ for action-level belief intervention."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game.engine import GameEngine, Observation
from game.constants import FOLD, CALL, RAISE
from game.evaluator import (
    HAND_STRENGTH_WEAK,
    HAND_STRENGTH_STRONG,
    opponent_hand_strength,
    HAND_STRENGTH_SAMPLES,
)

GATING_INPUT_DIM = 14
DEFAULT_GATING_PATH = "train/results/policy/belief_gating.pt"


class BeliefGatingNet(nn.Module):

    def __init__(self, input_dim: int = GATING_INPUT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pot_odds_and_ev(obs: Observation) -> tuple[float, float, float]:
    call_amount = max(0, obs.current_bet - obs.player_round_bet)
    pot_total = obs.pot + call_amount
    pot_odds = call_amount / pot_total if pot_total > 0 else 0.0
    equity = obs.equity if hasattr(obs, "equity") else 0.5
    ev_margin = equity - pot_odds
    return pot_odds, equity, ev_margin


def _opp_street_flags(opp_actions: list, current_round: int) -> tuple[float, float]:
    current = [(r, a) for r, a in opp_actions if r == current_round]
    raised = 1.0 if any(a == RAISE for _, a in current) else 0.0
    called = 1.0 if any(a == CALL for _, a in current) else 0.0
    return raised, called


def opp_aggression_score(opp_actions: list) -> float:
    if not opp_actions:
        return 0.0
    raises = sum(1 for _, a in opp_actions if a == RAISE)
    return min(raises / max(len(opp_actions), 1) * 2.0, 1.0)


def line_inconsistency_score(opp_actions: list, current_round: int,
                              belief_probs: np.ndarray) -> float:
    aggression = opp_aggression_score(opp_actions)
    weak_belief = float(belief_probs[0])
    raised, _ = _opp_street_flags(opp_actions, current_round)
    street_raise = 1.0 if raised > 0 else 0.0
    return min(aggression * weak_belief * (0.5 + 0.5 * street_raise), 1.0)


def build_gating_features(
    base_logits: np.ndarray,
    belief_probs: np.ndarray,
    uncertainty: float,
    obs: Observation,
    opp_actions: list,
) -> np.ndarray:
    pot_odds, equity, ev_margin = _pot_odds_and_ev(obs)
    aggression = opp_aggression_score(opp_actions)
    line_score = line_inconsistency_score(
        opp_actions, obs.current_round, belief_probs)
    opp_raised, opp_called = _opp_street_flags(opp_actions, obs.current_round)
    return np.array([
        base_logits[0], base_logits[1], base_logits[2],
        belief_probs[0], belief_probs[1], belief_probs[2],
        uncertainty, aggression, line_score,
        equity, pot_odds, ev_margin, opp_raised, opp_called,
    ], dtype=np.float32)


def oracle_exploit_action(
    base_action: int,
    legal: list[int],
    obs: Observation,
    opp_true_strength: float,
    opp_actions: list,
) -> int:
    pot_odds, equity, ev_margin = _pot_odds_and_ev(obs)
    opp_raised, opp_called = _opp_street_flags(opp_actions, obs.current_round)

    weak = opp_true_strength < HAND_STRENGTH_WEAK
    strong = opp_true_strength > HAND_STRENGTH_STRONG

    if weak and opp_raised > 0 and ev_margin > 0.05 and CALL in legal:
        return CALL
    if strong and opp_called > 0 and opp_raised == 0 and ev_margin < -0.05 and FOLD in legal:
        return FOLD
    return base_action if base_action in legal else legal[0]


def should_apply_gate(
    belief_probs: np.ndarray,
    uncertainty: float,
    obs: Observation,
    opp_actions: list,
    uncertainty_max: float = 0.35,
) -> bool:
    opp_raised, opp_called = _opp_street_flags(opp_actions, obs.current_round)
    _, _, ev_margin = _pot_odds_and_ev(obs)
    weak_belief = float(belief_probs[0])
    strong_belief = float(belief_probs[2])
    if uncertainty > uncertainty_max:
        return False
    if opp_raised > 0 and weak_belief > 0.22 and ev_margin > 0.0:
        return True
    if opp_called > 0 and opp_raised == 0 and strong_belief > 0.22 and ev_margin < 0.0:
        return True
    aggression = opp_aggression_score(opp_actions)
    if aggression > 0.35 and (weak_belief > 0.3 or strong_belief > 0.3):
        return True
    return False


def apply_learned_gating(
    gating_net: BeliefGatingNet,
    base_logits: np.ndarray,
    belief_probs: np.ndarray,
    uncertainty: float,
    legal: list[int],
    obs: Observation,
    opp_actions: list,
    device: str = "cpu",
    log: list | None = None,
    gate_scale: float = 1.0,
    selective: bool = False,
) -> np.ndarray:
    """
    Apply trained gate: logits' = base + delta, mask illegal, softmax.

    If log is provided, append detection metadata for bluff eval.
    selective: skip gate adjustment unless exploit context detected.
    gate_scale: scale residual delta ( <1 dampens gate when base policy is strong).
    """
    base_probs = _softmax_masked(base_logits, legal)
    if selective and not should_apply_gate(belief_probs, uncertainty, obs, opp_actions):
        if log is not None:
            log.append({"round": obs.current_round, "bluff_raise": False,
                        "slow_play_trap": False, "gate_skipped": True})
        return base_probs.astype(np.float32)

    feats = build_gating_features(
        base_logits, belief_probs, uncertainty, obs, opp_actions)
    x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    gating_net.eval()
    with torch.no_grad():
        delta = gating_net(x).squeeze(0).cpu().numpy()
    adjusted = base_logits.astype(np.float64) + gate_scale * delta

    gated_probs = _softmax_masked(adjusted, legal)

    if log is not None:
        opp_raised, opp_called = _opp_street_flags(opp_actions, obs.current_round)
        _, _, ev_margin = _pot_odds_and_ev(obs)
        bluff_raise = (
            opp_raised > 0
            and gated_probs[CALL] - base_probs[CALL] > 0.08
            and float(belief_probs[0]) > 0.25
        )
        slow_play_trap = (
            opp_called > 0 and opp_raised == 0
            and gated_probs[FOLD] - base_probs[FOLD] > 0.08
            and float(belief_probs[2]) > 0.25
        )
        log.append({
            "round": obs.current_round,
            "bluff_raise": bluff_raise,
            "slow_play_trap": slow_play_trap,
            "gate_skipped": False,
            "belief_weak": float(belief_probs[0]),
            "belief_medium": float(belief_probs[1]),
            "belief_strong": float(belief_probs[2]),
            "uncertainty": uncertainty,
            "equity": obs.equity,
            "ev_margin": ev_margin,
            "gate_delta": delta.tolist(),
        })

    return gated_probs.astype(np.float32)


def _softmax_masked(logits: np.ndarray, legal: list[int]) -> np.ndarray:
    masked = np.array([logits[a] if a in legal else -1e9 for a in range(3)])
    ex = np.exp(masked - masked.max())
    total = ex.sum()
    if total < 1e-12:
        probs = np.zeros(3)
        for a in legal:
            probs[a] = 1.0 / len(legal)
        return probs
    return ex / total


def logits_from_probs(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return np.log(p)


def logits_from_q_values(q_vals: list[float]) -> np.ndarray:
    return np.array(q_vals, dtype=np.float32)


def collect_gating_data(
    policy_agent,
    num_hands: int = 8000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect (gating_features, oracle_action) from policy vs Aggressive.

    policy_agent: L3Agent with loaded policy + belief (player 0).
    """
    from agents.aggressive_agent import AggressiveAgent

    samples_x: list[np.ndarray] = []
    samples_y: list[int] = []

    opponent = AggressiveAgent(name="Aggressive")
    engine = GameEngine(policy_agent, opponent)
    pid = policy_agent.player_id

    for hand_i in range(num_hands):
        policy_agent.reset()
        obs = engine.reset_hand()
        done = False
        step = 0

        while not done:
            step += 1
            if step > 60:
                break
            cp = engine.current_player

            if cp == pid:
                fb = policy_agent._feat_builder
                opp_actions = list(fb._opp_actions)

                features, belief_probs, uncertainty = policy_agent._build_policy_features(obs)
                x_pol = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(policy_agent.device)
                policy_agent.policy_net.eval()
                with torch.no_grad():
                    base_logits = policy_agent.policy_net(x_pol).squeeze(0).cpu().numpy()

                legal = obs.legal_actions
                base_action = int(base_logits.argmax())
                if base_action not in legal:
                    base_action = legal[0]

                opp_hole = engine.players[1 - pid].hole_cards
                opp_strength = opponent_hand_strength(
                    opp_hole, obs.community_cards, HAND_STRENGTH_SAMPLES)
                oracle = oracle_exploit_action(
                    base_action, legal, obs, opp_strength, opp_actions)

                samples_x.append(build_gating_features(
                    base_logits, belief_probs, uncertainty, obs, opp_actions))
                samples_y.append(oracle)

            action = engine.agents[cp].act(obs)
            policy_agent.record_action(cp, action, obs.current_round)
            obs, reward, done, info = engine.step(action)

        if verbose and (hand_i + 1) % 2000 == 0:
            print(f"  Gating data: {hand_i + 1}/{num_hands} hands, "
                  f"{len(samples_x)} decision samples")

    X = np.stack(samples_x) if samples_x else np.zeros((0, GATING_INPUT_DIM))
    y = np.array(samples_y, dtype=np.int64)
    return X, y


def collect_gating_data_from_q_agent(
    agent,
    num_hands: int = 8000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect gating training data for tabular L2 (BeliefSarsaAgent)."""
    from agents.aggressive_agent import AggressiveAgent

    samples_x: list[np.ndarray] = []
    samples_y: list[int] = []

    opponent = AggressiveAgent(name="Aggressive")
    engine = GameEngine(agent, opponent)

    for hand_i in range(num_hands):
        agent.reset()
        obs = engine.reset_hand()
        done = False
        step = 0

        while not done:
            step += 1
            if step > 60:
                break
            cp = engine.current_player
            if cp == agent.player_id:
                state = agent._encode_state(obs)
                q_vals = agent.q_table[state]
                base_logits = logits_from_q_values(q_vals)
                legal = obs.legal_actions
                base_action = max(legal, key=lambda a: q_vals[a])

                if agent.bnn_trained:
                    belief_probs, uncertainty = agent._predict_proba(obs)
                else:
                    belief_probs = np.ones(3) / 3.0
                    uncertainty = 1.0

                opp_hole = engine.players[1].hole_cards
                opp_strength = opponent_hand_strength(
                    opp_hole, obs.community_cards, HAND_STRENGTH_SAMPLES)

                oracle = oracle_exploit_action(
                    base_action, legal, obs, opp_strength, agent._opp_actions)

                gate_feats = build_gating_features(
                    base_logits, belief_probs, uncertainty, obs, agent._opp_actions)
                samples_x.append(gate_feats)
                samples_y.append(oracle)

            action = engine.agents[cp].act(obs)
            if hasattr(agent, "record_action"):
                agent.record_action(cp, action, obs.current_round)
            obs, reward, done, info = engine.step(action)

        if verbose and (hand_i + 1) % 2000 == 0:
            print(f"  Gating data (Q-agent): {hand_i + 1}/{num_hands}, "
                  f"{len(samples_x)} samples")

    X = np.stack(samples_x) if samples_x else np.zeros((0, GATING_INPUT_DIM))
    y = np.array(samples_y, dtype=np.int64)
    return X, y


def train_gating_net(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: str = DEFAULT_GATING_PATH,
    exploit_oversample: int = 5,
    init_path: str | None = None,
) -> BeliefGatingNet:
    """Train BeliefGatingNet with cross-entropy on oracle actions."""
    if len(X) < 64:
        raise ValueError(f"Need >= 64 gating samples, got {len(X)}")

    base_actions = X[:, :3].argmax(axis=1)
    exploit_mask = y != base_actions
    n_exploit = int(exploit_mask.sum())
    print(f"  Exploit labels (oracle ≠ base): {n_exploit}/{len(y)} "
          f"({100*n_exploit/len(y):.1f}%)")

    idx = np.arange(len(y))
    if n_exploit > 0 and exploit_oversample > 1:
        exploit_idx = idx[exploit_mask]
        extra = np.tile(exploit_idx, exploit_oversample - 1)
        idx = np.concatenate([idx, extra])
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    model = BeliefGatingNet().to(device)
    if init_path and Path(init_path).exists():
        ckpt = torch.load(init_path, map_location=device)
        model.load_state_dict(ckpt["gating_state_dict"])
        print(f"  Warm-started gate from {init_path}")
    base_logits = torch.tensor(X[:, :3], dtype=torch.float32)
    rest = torch.tensor(X[:, 3:], dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(base_logits, rest, y_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = 1.0 / (counts + 1)
    weights[0] *= 3.0  # upweight FOLD (bluff catch)
    weights[2] *= 2.0  # upweight RAISE
    weights = weights / weights.sum() * 3
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        for bl, rest_b, yb in loader:
            bl, rest_b, yb = bl.to(device), rest_b.to(device), yb.to(device)
            feats = torch.cat([bl, rest_b], dim=1)
            delta = model(feats)
            logits = bl + delta
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Gate epoch {epoch + 1}/{epochs}: "
                  f"loss={total_loss / n:.4f} acc={correct / n:.3f}")

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gating_state_dict": model.state_dict(), "input_dim": GATING_INPUT_DIM},
               path)
    print(f"[BeliefGatingNet] Saved to {path}")
    return model


def load_gating_net(path: str, device: str = "cpu") -> BeliefGatingNet:
    ckpt = torch.load(path, map_location=device)
    model = BeliefGatingNet(input_dim=ckpt.get("input_dim", GATING_INPUT_DIM))
    model.load_state_dict(ckpt["gating_state_dict"])
    model.to(device)
    model.eval()
    return model
