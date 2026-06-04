# train/train_nn_mc_distill.py — NN_MC Training with SARSA Policy Distillation
"""
SARSA → BNN Policy Distillation pipeline for 52-card Texas Hold'em.

KEY INNOVATION: Instead of collecting BNN labels from opponent's true hand
strength (only at showdown → ~1 sample/hand), we use SARSA's greedy action
at EVERY decision point as a proxy label:
    RAISE → opponent WEAK   (0)   "I think I'm ahead"
    CALL  → opponent MID    (1)   "EV is marginal"
    FOLD  → opponent STRONG (2)   "I'm far behind"

This yields 5-10× more samples and doesn't require showdown.

Pipeline:
  Phase 1a — Load trained SARSA, collect distillation data (SARSA ε=0 vs Expert)
  Phase 1b — Train BNN on distilled data (mask_prob=0.5, ~200 epochs)
  Phase 2  — NN_MC SARSA Q-table training (vs Expert, gated BNN state + online finetune)

Usage:
    python -u train/train_nn_mc_distill.py [num_hands] [output_path] \\
        [distill_hands] [sarsa_model]

Example:
    python -u train/train_nn_mc_distill.py 50000 train/nn_mc_distill.pt \\
        20000 train/sarsa_final.pkl
"""
from __future__ import annotations

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from game.engine import GameEngine
from agents.nn_mc_agent import (
    NN_MCAgent, BNNWithMCDropout,
    collect_bnn_data_sarsa_distill, train_bnn, _equity_to_strength_label,
)
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent


# =========================================================================
#  Phase 1: SARSA Distillation → Train BNN
# =========================================================================

def phase1_distill_and_train_bnn(
    distill_hands=20000,
    bnn_epochs=200,
    bnn_batch_size=64,
    bnn_lr=1e-3,
    sarsa_model_path="train/sarsa_final.pkl",
    bnn_save_path="train/bnn_distilled.pt",
    mask_prob=0.5,
):
    print("=" * 60)
    print("  Phase 1: SARSA Policy Distillation → BNN Training")
    print(f"  SARSA model:     {sarsa_model_path}")
    print(f"  Distill hands:   {distill_hands}")
    print(f"  Feature masking: {mask_prob:.0%}")
    print(f"  BNN epochs:      {bnn_epochs}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # --- Load trained SARSA agent (ε=0 for pure exploitation) ---
    print(f"\n  Loading SARSA agent from {sarsa_model_path}...")
    sarsa_agent = SarsaAgent(name="SARSA_teacher", epsilon=0.0,
                             load_q_table_path=sarsa_model_path)
    q_size = len(sarsa_agent.q_table)
    print(f"  SARSA Q-table size: {q_size} states")

    # --- Collect distillation data ---
    print(f"\n  Collecting distillation data ({distill_hands} hands)...")
    t0 = time.time()
    expert = ExpertAgent(name="Expert")
    env = GameEngine(sarsa_agent, expert)

    X, y, mask_flags = collect_bnn_data_sarsa_distill(
        env, sarsa_agent, num_hands=distill_hands,
        mask_prob=mask_prob, verbose=True, observer_player=0)

    elapsed = time.time() - t0
    print(f"\n  Collected {len(X)} samples in {elapsed:.1f}s "
          f"({len(X)/distill_hands:.1f} samples/hand, "
          f"~{len(X)/max(distill_hands,1):.0f}× vs showdown-only)")
    print(f"  Masked: {np.sum(mask_flags)}/{len(X)} = {np.mean(mask_flags):.1%}")
    print(f"  Label distribution: weak={np.sum(y==0)} "
          f"({np.sum(y==0)/len(y)*100:.1f}%), "
          f"mid={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%), "
          f"strong={np.sum(y==2)} ({np.sum(y==2)/len(y)*100:.1f}%)")

    # --- Train BNN ---
    print(f"\n  Training BNN ({bnn_epochs} epochs, val_split=0.2)...")
    model = BNNWithMCDropout(input_dim=47).to(device)
    model = train_bnn(model, X, y, mask_flags=mask_flags, epochs=bnn_epochs,
                      batch_size=bnn_batch_size, lr=bnn_lr,
                      val_split=0.2, device=device, verbose=True)

    torch.save({
        "bnn_state_dict": model.state_dict(),
        "bnn_trained": True,
        "distill_samples": len(X),
        "sarsa_model": sarsa_model_path,
    }, bnn_save_path)
    print(f"\n  Distilled BNN saved to {bnn_save_path}")
    return model, device


# =========================================================================
#  Phase 2: NN_MC SARSA Q-table Training (same as train_nn_mc.py)
# =========================================================================

def train_one_hand_sarsa_bnn(env, agent, agent_id=0):
    """SARSA TD-update with BNN-augmented state (gated mode)."""
    obs = env.reset_hand()
    agent.reset()
    done = False
    step_count = 0

    prev_state = None
    prev_action = None
    agent_reward = 0.0

    while not done:
        step_count += 1
        if step_count > 50:
            break

        cp = env.current_player

        if cp == agent_id:
            state = agent._encode_state(obs)
            action = agent.act(obs)

            if prev_state is not None:
                agent.learn_sarsa(prev_state, prev_action, 0.0,
                                  state, action, done=False)

            round_before = obs.current_round
            obs, reward, done, info = env.step(action)
            agent.record_action(cp, action, round_before)

            if done:
                agent_reward = info.get("result").rewards[agent_id]
                agent.learn_sarsa(state, action, agent_reward,
                                  None, None, done=True)
            else:
                prev_state, prev_action = state, action
        else:
            round_before = obs.current_round
            opp_action = env.agents[cp].act(obs)
            obs, reward, done, info = env.step(opp_action)
            agent.record_action(cp, opp_action, round_before)

            if done and prev_state is not None:
                agent_reward = info.get("result").rewards[agent_id]
                agent.learn_sarsa(prev_state, prev_action, agent_reward,
                                  None, None, done=True)

    agent.decay_epsilon()
    return agent_reward


def _compute_gated_stats(agent: NN_MCAgent) -> float:
    """Return fraction of Q-table states that are BNN-augmented."""
    if len(agent.q_table) == 0:
        return 0.0
    gated = sum(1 for s in agent.q_table if len(s) >= 4 and s[3] != -1)
    return gated / len(agent.q_table)


def _finetune_bnn_online(agent: NN_MCAgent, X: np.ndarray, y: np.ndarray,
                          device: str, epochs: int = 5, lr: float = 5e-5):
    """Lightweight online fine-tuning of BNN on Phase 2 data."""
    model = agent.bnn_model
    model.train()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    class_counts = np.bincount(y, minlength=model.num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * model.num_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def phase2_train_sarsa_qtable(
    bnn_model, device, num_hands=50000,
    bnn_model_path="train/bnn_distilled.pt",
    output_path="train/nn_mc_distill.pt",
    state_mode='gated',
    bnn_finetune_interval=5000,
):
    print("\n" + "=" * 60)
    print("  Phase 2: SARSA Q-table Training (gated BNN state + TD learning)")
    print("=" * 60)
    print(f"  Total hands:  {num_hands}")
    print(f"  State mode:   {state_mode}")
    print(f"  BNN source:   {bnn_model_path}")
    print(f"  ε_min:        0.05")
    print(f"  BNN finetune: every {bnn_finetune_interval} hands")
    print(f"  Output:       {output_path}")

    agent = NN_MCAgent(
        name="NN_MC_distill",
        epsilon=1.0, epsilon_decay=0.9998, epsilon_min=0.05,
        alpha=0.1, gamma=0.95, mc_samples=20, device=device,
        state_mode=state_mode,
    )
    agent.bnn_model = bnn_model
    agent.bnn_trained = True
    agent._auto_record_self = False

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    ft_X, ft_y = [], []

    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000

    for hand in range(1, num_hands + 1):
        r = train_one_hand_sarsa_bnn(env, agent, agent_id=0)
        chips_window += r
        if r > 0:
            wins_window += 1

        # Collect BNN fine-tuning data from showdown hands
        if agent.bnn_trained and hasattr(env, 'players'):
            try:
                opp = env.players[1]
                if not opp.folded and len(env.community_cards) >= 3:
                    obs_for_ft = env._get_observation(0)
                    opp_hole = opp.hole_cards
                    if len(opp_hole) == 2:
                        from treys import Card
                        opp_ranks = [Card.get_rank_int(c) for c in opp_hole]
                        opp_rank_avg = sum(opp_ranks) / (len(opp_ranks) * 12.0)
                        opp_suited = 1.0 if Card.get_suit_int(opp_hole[0]) == Card.get_suit_int(opp_hole[1]) else 0.0
                        from game.evaluator import compute_equity
                        opp_eq = compute_equity(opp_hole, env.community_cards)
                        label = _equity_to_strength_label(opp_eq)
                        feat = agent._encode_bnn_features(obs_for_ft)
                        ft_X.append(feat)
                        ft_y.append(label)
            except Exception:
                pass

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_chips = chips_window / window_size
            wr = wins_window / window_size
            gated_pct = _compute_gated_stats(agent) if state_mode == 'gated' else 0
            print(
                f"| Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                f"Qsize={agent.get_q_table_size():>5} | "
                f"AvgChips={avg_chips:+.1f} | "
                f"WR={wr:.1%} | time={elapsed:.1f}s"
                + (f" | gated={gated_pct:.0%}" if state_mode == 'gated' else "")
            )
            chips_window = 0.0
            wins_window = 0

        if bnn_finetune_interval > 0 and hand % bnn_finetune_interval == 0 and len(ft_X) >= 200:
            n_ft = min(len(ft_X), 3000)
            X_ft = np.array(ft_X[-n_ft:], dtype=np.float32)
            y_ft = np.array(ft_y[-n_ft:], dtype=np.int64)
            _finetune_bnn_online(agent, X_ft, y_ft, device, epochs=5, lr=5e-5)
            ft_X = ft_X[-2000:]
            ft_y = ft_y[-2000:]

    agent.save_model(output_path)
    total_time = time.time() - start
    print(f"\nPhase 2 completed in {total_time:.1f}s ({total_time / 60:.1f}min).")
    print(f"Final Q-table size: {agent.get_q_table_size()}")


# =========================================================================
#  Main
# =========================================================================

def main():
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "train/nn_mc_distill.pt"
    distill_hands = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    sarsa_model = sys.argv[4] if len(sys.argv) > 4 else "train/sarsa_final.pkl"
    bnn_path = sys.argv[5] if len(sys.argv) > 5 else "train/bnn_distilled.pt"
    # 5th optional arg: "--skip_phase1" to load pre-trained BNN
    skip_phase1 = any(a == "--skip_phase1" for a in sys.argv)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if skip_phase1 and os.path.exists(bnn_path):
        # Resume: load pre-trained BNN, skip to Phase 2
        print("=" * 60)
        print("  RESUME MODE: Loading pre-trained BNN, skipping Phase 1")
        print(f"  BNN model: {bnn_path}")
        print("=" * 60)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BNNWithMCDropout(input_dim=47).to(device)
        checkpoint = torch.load(bnn_path, map_location=device)
        model.load_state_dict(checkpoint["bnn_state_dict"])
        print(f"  BNN loaded (trained={checkpoint.get('bnn_trained', True)})")
        bnn_model = model
    else:
        # Phase 1: SARSA distillation → train BNN
        bnn_model, device = phase1_distill_and_train_bnn(
            distill_hands=distill_hands,
            bnn_epochs=200,
            sarsa_model_path=sarsa_model,
            bnn_save_path=bnn_path,
            mask_prob=0.5,
        )

    # Phase 2: NN_MC SARSA Q-table training
    phase2_train_sarsa_qtable(
        bnn_model=bnn_model,
        device=device,
        num_hands=num_hands,
        bnn_model_path=bnn_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
