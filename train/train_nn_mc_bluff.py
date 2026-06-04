# train/train_nn_mc_bluff.py — NN_MC training with bluff-rich multi-source BNN data
"""
Enhanced training pipeline: BNN learns from diverse opponent styles.

Key improvement over train_nn_mc.py:
  - Phase 1 collects data from MULTIPLE match-ups:
      1. Expert vs Expert (Nash equilibrium bluffing on both sides)
      2. Expert vs Aggressive (bluff detection training)
      3. SARSA vs Expert (existing pattern)
  - This gives BNN exposure to different bluffing styles & frequencies
  - Phase 2 uses compact state space for faster Q-table convergence

Usage:
    conda activate fmd
    python train/train_nn_mc_bluff.py [sarsa_hands] [output_path]
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from game.engine import GameEngine
from agents.nn_mc_agent import (
    NN_MCAgent, BNNWithMCDropout,
    collect_bnn_training_data, train_bnn,
)
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent
from agents.aggressive_agent import AggressiveAgent, TightPassiveAgent
from train.train_nn_mc import train_one_hand_sarsa_bnn


# =========================================================================
#  Phase 1: Multi-source BNN data collection + training
# =========================================================================

def phase1_multi_source_bnn(
    bnn_epochs=200,
    bnn_batch_size=64,
    bnn_lr=1e-3,
    mask_prob=0.5,
    sarsa_model_path="train/sarsa_final.pkl",
    bnn_save_path="train/bnn_pretrained.pt",
    bnn_hidden_dims=(256, 128, 64),
    bnn_dropout=0.2,
    hands_per_source=7000,
):
    """
    Collect BNN training data from multiple match-ups:
      1. Expert(p0) vs Expert(p1): balanced Nash play, optimal bluffing
      2. Expert(p0) vs Aggressive(p1): observer sees aggressive bluffs
      3. SARSA(p0) vs Expert(p1): conservative observer vs Nash bluffer
      4. Expert(p0) vs TightPassive(p1): observer sees predictable play (contrast)

    All sources predict player 1's hand strength from player 0's perspective.
    """
    print("=" * 70)
    print("  Phase 1: Multi-Source BNN Pre-training (Bluff-Rich Data)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Mask prob: {mask_prob:.0%}")
    print(f"  BNN architecture: {bnn_hidden_dims}, dropout={bnn_dropout}")
    print(f"  Hands per source: {hands_per_source}")

    all_X, all_y, all_mask = [], [], []
    t0 = time.time()

    # --- Source 1: Expert vs Expert (Nash bluffing both sides) ---
    print(f"\n  [Source 1] Expert vs Expert ({hands_per_source} hands)...")
    expert0 = ExpertAgent(name="Expert_p0")
    expert1 = ExpertAgent(name="Expert_p1")
    env1 = GameEngine(expert0, expert1)
    X1, y1, m1 = collect_bnn_training_data(
        env1, num_hands=hands_per_source, mask_prob=mask_prob,
        verbose=True, target_player=1)
    all_X.append(X1); all_y.append(y1); all_mask.append(m1)
    print(f"    → {len(X1)} samples")

    # --- Source 2: Expert vs Aggressive (bluff-heavy opponent) ---
    print(f"\n  [Source 2] Expert vs Aggressive ({hands_per_source} hands)...")
    expert_obs = ExpertAgent(name="Expert_obs")
    aggressive = AggressiveAgent(name="Aggressive_p1",
                                 bluff_raise_prob=0.45,
                                 value_raise_prob=0.80)
    env2 = GameEngine(expert_obs, aggressive)
    X2, y2, m2 = collect_bnn_training_data(
        env2, num_hands=hands_per_source, mask_prob=mask_prob,
        verbose=True, target_player=1)
    all_X.append(X2); all_y.append(y2); all_mask.append(m2)
    print(f"    → {len(X2)} samples")

    # --- Source 3: SARSA vs Expert (conservative observer) ---
    print(f"\n  [Source 3] SARSA vs Expert ({hands_per_source} hands)...")
    sarsa_agent = SarsaAgent(name="SARSA_p0", epsilon=0.0,
                             load_q_table_path=sarsa_model_path)
    expert_opp = ExpertAgent(name="Expert_p1_s3")
    env3 = GameEngine(sarsa_agent, expert_opp)
    X3, y3, m3 = collect_bnn_training_data(
        env3, num_hands=hands_per_source, mask_prob=mask_prob,
        verbose=True, target_player=1)
    all_X.append(X3); all_y.append(y3); all_mask.append(m3)
    print(f"    → {len(X3)} samples")

    # --- Source 4: Expert vs TightPassive (contrast data) ---
    print(f"\n  [Source 4] Expert vs TightPassive ({hands_per_source} hands)...")
    expert_obs2 = ExpertAgent(name="Expert_obs2")
    tight = TightPassiveAgent(name="Tight_p1")
    env4 = GameEngine(expert_obs2, tight)
    X4, y4, m4 = collect_bnn_training_data(
        env4, num_hands=hands_per_source, mask_prob=mask_prob,
        verbose=True, target_player=1)
    all_X.append(X4); all_y.append(y4); all_mask.append(m4)
    print(f"    → {len(X4)} samples")

    # --- Merge all sources ---
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    mask_flags = np.concatenate(all_mask, axis=0)
    elapsed = time.time() - t0

    print(f"\n  Total: {len(X)} samples from 4 sources in {elapsed:.1f}s")
    print(f"  Masked: {np.mean(mask_flags):.1%}")
    print(f"  Labels: weak={np.sum(y==0)}, mid={np.sum(y==1)}, strong={np.sum(y==2)}")

    # --- Train BNN ---
    print(f"\n  Training BNN ({bnn_epochs} epochs)...")
    model = BNNWithMCDropout(
        input_dim=47,
        hidden_dims=bnn_hidden_dims,
        dropout_rate=bnn_dropout,
    ).to(device)
    model = train_bnn(model, X, y, mask_flags=mask_flags,
                      epochs=bnn_epochs, batch_size=bnn_batch_size,
                      lr=bnn_lr, val_split=0.2, device=device, verbose=True)

    torch.save({"bnn_state_dict": model.state_dict(), "bnn_trained": True}, bnn_save_path)
    print(f"  BNN saved to {bnn_save_path}")
    return model, device


# =========================================================================
#  Phase 2: SARSA Q-table with compact state
# =========================================================================

def phase2_compact_sarsa(
    bnn_model, device, num_hands=50000,
    bnn_model_path="train/bnn_pretrained.pt",
    output_path="train/nn_mc_model.pt",
    state_mode='compact',
    bnn_hidden_dims=(256, 128, 64),
    bnn_dropout=0.2,
):
    print("\n" + "=" * 70)
    print("  Phase 2: SARSA Q-table (compact state + bluff-trained BNN)")
    print("=" * 70)
    print(f"  State mode: {state_mode}")
    print(f"  Total hands: {num_hands}")
    print(f"  Output: {output_path}")

    agent = NN_MCAgent(
        name="NN_MC_Bluff",
        epsilon=1.0, epsilon_decay=0.9998, epsilon_min=0.10,
        alpha=0.1, gamma=0.95, mc_samples=20, device=device,
        state_mode=state_mode,
        bnn_hidden_dims=bnn_hidden_dims,
        bnn_dropout=bnn_dropout,
    )
    agent.bnn_model = bnn_model
    agent.bnn_trained = True
    agent._auto_record_self = False

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    start = time.time()
    total_reward_window = 0.0
    wins_window = 0
    window_size = 1000

    for hand in range(1, num_hands + 1):
        r = train_one_hand_sarsa_bnn(env, agent, agent_id=0)
        total_reward_window += r
        if r > 0:
            wins_window += 1

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_r = total_reward_window / window_size
            wr = wins_window / window_size
            print(
                f"| Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                f"Qsize={agent.get_q_table_size():>5} | "
                f"AvgR(last{window_size})={avg_r:+.2f} | "
                f"WR={wr:.1%} | time={elapsed:.1f}s"
            )
            total_reward_window = 0.0
            wins_window = 0

    agent.save_model(output_path)
    total_time = time.time() - start
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f}min).")
    print(f"Final Q-table size: {agent.get_q_table_size()}")


# =========================================================================
#  Main
# =========================================================================

def main():
    sarsa_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    state_mode = sys.argv[3] if len(sys.argv) > 3 else 'compact'

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"train/results/bluff_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(out_dir, "nn_mc_vs_expert.pt")

    bnn_path = os.path.join(out_dir, "bnn_pretrained.pt")

    print("=" * 70)
    print("  NN_MC BLUFF-ENRICHED TRAINING")
    print(f"  State mode: {state_mode}")
    print(f"  SARSA hands: {sarsa_hands}")
    print(f"  Output dir: {out_dir}")
    print("=" * 70)

    bnn_model, device = phase1_multi_source_bnn(
        bnn_epochs=200,
        mask_prob=0.5,
        bnn_save_path=bnn_path,
        bnn_hidden_dims=(256, 128, 64),
        bnn_dropout=0.2,
        hands_per_source=7000,  # 7k × 4 sources = 28k hands → ~100k samples
    )

    phase2_compact_sarsa(
        bnn_model=bnn_model,
        device=device,
        num_hands=sarsa_hands,
        bnn_model_path=bnn_path,
        output_path=output_path,
        state_mode=state_mode,
        bnn_hidden_dims=(256, 128, 64),
        bnn_dropout=0.2,
    )


if __name__ == "__main__":
    main()
