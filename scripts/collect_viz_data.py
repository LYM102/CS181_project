#!/usr/bin/env python3
"""Collect per-hand, per-decision-point data for all visualization experiments.

Outputs (in logs/viz_data/):
  l1_vs_agg.csv  — L1 (Belief SARSA) vs Aggressive
  l2_vs_agg.csv  — L2 (Tabular + Gate) vs Aggressive
  l3_vs_agg.csv  — L3 (Neural + Gate) vs Aggressive
  l0_vs_agg.csv  — L0 (SARSA baseline) vs Aggressive
"""

import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from game.engine import GameEngine
from game.constants import FOLD, CALL, RAISE
from game.match_eval import reset_eval_stacks
from game.evaluator import (
    opponent_hand_strength,
    hand_strength_to_label,
    HAND_STRENGTH_SAMPLES,
)
from agents.aggressive_agent import AggressiveAgent
from agents.belief_gating import apply_learned_gating, logits_from_q_values

# Reuse loaders from eval_progressive
from eval_progressive import (
    load_l0, load_l1, load_l2, load_l3,
    TrackedAggressive, CFR_POLICY,
)

SEED = 42
NUM_HANDS = 3000
OUT_DIR = PROJECT / "logs" / "viz_data"
ACTION_NAMES = {0: "Fold", 1: "Call", 2: "Raise"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_opp_true_strength(engine, pid):
    """Get opponent true hand strength label at showdown.
    Returns -1 if opponent folded (no showdown)."""
    opp_id = 1 - pid
    opp = engine.players[opp_id]
    if opp.folded:
        return -1
    if len(engine.community_cards) < 3:
        return -1
    strength = opponent_hand_strength(
        opp.hole_cards, engine.community_cards, HAND_STRENGTH_SAMPLES)
    return hand_strength_to_label(strength)


def _get_hand_type(opp):
    """Extract hand_type from TrackedAggressive ground truth trace."""
    gt = opp.pop_gt()
    if any(g["is_bluff"] for g in gt):
        return 1  # bluff
    if any(g["is_trap"] for g in gt):
        return 2  # trap
    return 0  # normal


def _flush_hand(writer, buffer, hand_type, reward):
    """Write buffered rows with hand_type and reward filled in."""
    for row in buffer:
        row["hand_type"] = hand_type
        row["reward"] = reward
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Task 1: L1 vs Aggressive
# ---------------------------------------------------------------------------

def collect_l1(num_hands=NUM_HANDS, seed=SEED):
    print("=" * 60)
    print(f"Task 1: L1 vs Aggressive — {num_hands} hands")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = load_l1()
    opp = TrackedAggressive(policy_path=str(CFR_POLICY))
    pid = agent.player_id
    engine = GameEngine(agent, opp)

    out_path = OUT_DIR / "l1_vs_agg.csv"
    fieldnames = [
        "hand_id", "step", "round", "P_weak", "P_mid", "P_strong",
        "true_strength", "equity", "reward", "hand_type",
    ]

    total_rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hand_id in range(num_hands):
            reset_eval_stacks(engine)
            agent.reset()
            opp.reset()
            obs = engine.reset_hand()
            done = False
            step = 0
            buffer = []
            log_start = len(agent.bluff_log)

            while not done:
                step += 1
                if step > 60:
                    break
                cp = engine.current_player

                if cp == pid:
                    belief_probs, uncertainty = agent._predict_proba(obs)
                    row = {
                        "hand_id": hand_id,
                        "step": step,
                        "round": obs.current_round,
                        "P_weak": f"{belief_probs[0]:.6f}",
                        "P_mid": f"{belief_probs[1]:.6f}",
                        "P_strong": f"{belief_probs[2]:.6f}",
                        "true_strength": "",  # filled at hand end
                        "equity": f"{obs.equity:.6f}",
                    }
                    buffer.append(row)
                    action = agent.act(obs)
                else:
                    action = engine.agents[cp].act(obs)

                if hasattr(agent, "record_action"):
                    agent.record_action(cp, action, obs.current_round)
                obs, reward, done, info = engine.step(action)

            # Hand ended — get true strength and result
            true_str = _get_opp_true_strength(engine, pid)
            hand_type = _get_hand_type(opp)
            hand_reward = info["result"].rewards[pid] if "result" in info else reward

            for row in buffer:
                row["true_strength"] = true_str
            _flush_hand(writer, buffer, hand_type, hand_reward)
            total_rows += len(buffer)

            if (hand_id + 1) % 500 == 0:
                print(f"  [{hand_id + 1}/{num_hands}] rows={total_rows}")

    print(f"  Saved {total_rows} rows → {out_path}")


# ---------------------------------------------------------------------------
# Task 2: L3+Gate vs Aggressive
# ---------------------------------------------------------------------------

def collect_l3(num_hands=NUM_HANDS, seed=SEED):
    print("=" * 60)
    print(f"Task 2: L3+Gate vs Aggressive — {num_hands} hands")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = load_l3(belief_gate=True, gate_selective=True, gate_scale=0.5)
    opp = TrackedAggressive(policy_path=str(CFR_POLICY))
    pid = agent.player_id
    engine = GameEngine(agent, opp)

    out_path = OUT_DIR / "l3_vs_agg.csv"
    fieldnames = [
        "hand_id", "step", "round", "P_weak", "P_mid", "P_strong",
        "correction_magnitude", "gate_delta_0", "gate_delta_1", "gate_delta_2",
        "original_action", "final_action",
        "true_strength", "equity", "reward", "hand_type",
    ]

    total_rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hand_id in range(num_hands):
            reset_eval_stacks(engine)
            agent.reset()
            opp.reset()
            obs = engine.reset_hand()
            done = False
            step = 0
            buffer = []
            log_start = len(agent.bluff_log)

            while not done:
                step += 1
                if step > 60:
                    break
                cp = engine.current_player

                if cp == pid:
                    # Build features and get BNN prediction
                    features, belief_probs, uncertainty = agent._build_policy_features(obs)
                    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(agent.device)

                    agent.policy_net.eval()
                    with torch.no_grad():
                        base_logits = agent.policy_net(x).squeeze(0).cpu().numpy()

                    legal = obs.legal_actions

                    # Base action (without gate)
                    masked_logits = np.array([
                        base_logits[a] if a in legal else -1e9 for a in range(3)
                    ])
                    base_action = int(masked_logits.argmax())

                    # Record bluff_log position before gating
                    log_before = len(agent.bluff_log)

                    # Apply gating → final action
                    probs = agent._apply_belief_gating(
                        base_logits, belief_probs, uncertainty, legal, obs)
                    masked_probs = np.array([
                        probs[a] if a in legal else -1.0 for a in range(3)
                    ])
                    final_action = int(masked_probs.argmax())

                    # Extract gate delta from bluff_log
                    gate_delta = [0.0, 0.0, 0.0]
                    if len(agent.bluff_log) > log_before:
                        entry = agent.bluff_log[-1]
                        if "gate_delta" in entry:
                            gate_delta = entry["gate_delta"]

                    corr_mag = float(np.linalg.norm(gate_delta))

                    row = {
                        "hand_id": hand_id,
                        "step": step,
                        "round": obs.current_round,
                        "P_weak": f"{belief_probs[0]:.6f}",
                        "P_mid": f"{belief_probs[1]:.6f}",
                        "P_strong": f"{belief_probs[2]:.6f}",
                        "correction_magnitude": f"{corr_mag:.6f}",
                        "gate_delta_0": f"{gate_delta[0]:.6f}",
                        "gate_delta_1": f"{gate_delta[1]:.6f}",
                        "gate_delta_2": f"{gate_delta[2]:.6f}",
                        "original_action": base_action,
                        "final_action": final_action,
                        "true_strength": "",
                        "equity": f"{obs.equity:.6f}",
                    }
                    buffer.append(row)
                    action = final_action
                else:
                    action = engine.agents[cp].act(obs)

                agent.record_action(cp, action, obs.current_round)
                obs, reward, done, info = engine.step(action)

            true_str = _get_opp_true_strength(engine, pid)
            hand_type = _get_hand_type(opp)
            hand_reward = info["result"].rewards[pid] if "result" in info else reward

            for row in buffer:
                row["true_strength"] = true_str
            _flush_hand(writer, buffer, hand_type, hand_reward)
            total_rows += len(buffer)

            if (hand_id + 1) % 500 == 0:
                print(f"  [{hand_id + 1}/{num_hands}] rows={total_rows}")

    print(f"  Saved {total_rows} rows → {out_path}")


# ---------------------------------------------------------------------------
# Task 3: L2 vs Aggressive
# ---------------------------------------------------------------------------

def collect_l2(num_hands=NUM_HANDS, seed=SEED):
    print("=" * 60)
    print(f"Task 3: L2+Gate vs Aggressive — {num_hands} hands")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = load_l2()
    opp = TrackedAggressive(policy_path=str(CFR_POLICY))
    pid = agent.player_id
    engine = GameEngine(agent, opp)

    out_path = OUT_DIR / "l2_vs_agg.csv"
    fieldnames = [
        "hand_id", "step", "round", "P_weak", "P_mid", "P_strong",
        "correction_magnitude", "gate_delta_0", "gate_delta_1", "gate_delta_2",
        "original_action", "final_action",
        "true_strength", "equity", "reward", "hand_type",
    ]

    total_rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hand_id in range(num_hands):
            reset_eval_stacks(engine)
            agent.reset()
            opp.reset()
            obs = engine.reset_hand()
            done = False
            step = 0
            buffer = []
            log_start = len(agent.bluff_log)

            while not done:
                step += 1
                if step > 60:
                    break
                cp = engine.current_player

                if cp == pid:
                    # Get BNN prediction
                    belief_probs, uncertainty = agent._predict_proba(obs)
                    legal = obs.legal_actions

                    # Base action from Q-table (without gate)
                    state = agent._encode_state(obs)
                    q_vals = agent.q_table[state]
                    base_logits = logits_from_q_values(
                        np.array(q_vals, dtype=np.float32))
                    best_q = max(q_vals[a] for a in legal)
                    base_action = random.choice(
                        [a for a in legal if q_vals[a] == best_q])

                    # Record bluff_log before gating
                    log_before = len(agent.bluff_log)

                    # Apply gating
                    probs = apply_learned_gating(
                        agent.gating_net, base_logits, belief_probs, uncertainty,
                        legal, obs, agent._opp_actions, device=agent.device,
                        log=agent.bluff_log, gate_scale=agent.gate_scale,
                        selective=agent.gate_selective)

                    masked_probs = np.array([
                        probs[a] if a in legal else -1.0 for a in range(3)
                    ])
                    final_action = int(masked_probs.argmax())

                    # Extract gate delta
                    gate_delta = [0.0, 0.0, 0.0]
                    if len(agent.bluff_log) > log_before:
                        entry = agent.bluff_log[-1]
                        if "gate_delta" in entry:
                            gate_delta = entry["gate_delta"]

                    corr_mag = float(np.linalg.norm(gate_delta))

                    row = {
                        "hand_id": hand_id,
                        "step": step,
                        "round": obs.current_round,
                        "P_weak": f"{belief_probs[0]:.6f}",
                        "P_mid": f"{belief_probs[1]:.6f}",
                        "P_strong": f"{belief_probs[2]:.6f}",
                        "correction_magnitude": f"{corr_mag:.6f}",
                        "gate_delta_0": f"{gate_delta[0]:.6f}",
                        "gate_delta_1": f"{gate_delta[1]:.6f}",
                        "gate_delta_2": f"{gate_delta[2]:.6f}",
                        "original_action": base_action,
                        "final_action": final_action,
                        "true_strength": "",
                        "equity": f"{obs.equity:.6f}",
                    }
                    buffer.append(row)
                    action = final_action
                else:
                    action = engine.agents[cp].act(obs)

                if hasattr(agent, "record_action"):
                    agent.record_action(cp, action, obs.current_round)
                obs, reward, done, info = engine.step(action)

            true_str = _get_opp_true_strength(engine, pid)
            hand_type = _get_hand_type(opp)
            hand_reward = info["result"].rewards[pid] if "result" in info else reward

            for row in buffer:
                row["true_strength"] = true_str
            _flush_hand(writer, buffer, hand_type, hand_reward)
            total_rows += len(buffer)

            if (hand_id + 1) % 500 == 0:
                print(f"  [{hand_id + 1}/{num_hands}] rows={total_rows}")

    print(f"  Saved {total_rows} rows → {out_path}")


# ---------------------------------------------------------------------------
# Task 4: L0 vs Aggressive
# ---------------------------------------------------------------------------

def collect_l0(num_hands=NUM_HANDS, seed=SEED):
    print("=" * 60)
    print(f"Task 4: L0 (SARSA) vs Aggressive — {num_hands} hands")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)

    agent = load_l0()
    opp = TrackedAggressive(policy_path=str(CFR_POLICY))
    pid = 0  # SarsaAgent has no player_id; it is always player 0
    engine = GameEngine(agent, opp)

    out_path = OUT_DIR / "l0_vs_agg.csv"
    fieldnames = [
        "hand_id", "step", "round", "phi", "equity",
        "final_result", "reward", "hand_type",
    ]

    total_rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hand_id in range(num_hands):
            reset_eval_stacks(engine)
            agent.reset()
            opp.reset()
            obs = engine.reset_hand()
            done = False
            step = 0
            buffer = []

            while not done:
                step += 1
                if step > 60:
                    break
                cp = engine.current_player

                if cp == pid:
                    row = {
                        "hand_id": hand_id,
                        "step": step,
                        "round": obs.current_round,
                        "phi": f"{obs.equity:.6f}",
                        "equity": f"{obs.equity:.6f}",
                        "final_result": "",  # filled at hand end
                    }
                    buffer.append(row)
                    action = agent.act(obs)
                else:
                    action = engine.agents[cp].act(obs)

                obs, reward, done, info = engine.step(action)

            hand_reward = info["result"].rewards[pid] if "result" in info else reward
            # final_result: win=1, loss=-1, tie=0
            if "result" in info:
                res = info["result"]
                if res.winner == pid:
                    final_result = 1
                elif res.winner is None:
                    final_result = 0
                else:
                    final_result = -1
            else:
                final_result = 0

            hand_type = _get_hand_type(opp)

            for row in buffer:
                row["final_result"] = final_result
            _flush_hand(writer, buffer, hand_type, hand_reward)
            total_rows += len(buffer)

            if (hand_id + 1) % 500 == 0:
                print(f"  [{hand_id + 1}/{num_hands}] rows={total_rows}")

    print(f"  Saved {total_rows} rows → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    collect_l1()
    print()
    collect_l3()
    print()
    collect_l2()
    print()
    collect_l0()

    print("\n" + "=" * 60)
    print("All data collection complete!")
    print("=" * 60)

    # Summary
    for name in ["l1_vs_agg", "l3_vs_agg", "l2_vs_agg", "l0_vs_agg"]:
        p = OUT_DIR / f"{name}.csv"
        if p.exists():
            with open(p) as f:
                n = sum(1 for _ in f) - 1  # minus header
            print(f"  {name}.csv: {n} rows")


if __name__ == "__main__":
    main()
