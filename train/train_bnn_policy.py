# train/train_bnn_policy.py — BNN Policy via SARSA BC + DAgger Fine-tuning
"""
End-to-End Neural Policy with DAgger (Dataset Aggregation) fine-tuning.

Pipeline:
  Phase 1  - SARSA Behavioral Cloning: (feature, SARSA_action) -> CrossEntropy
  Phase 2  - DAgger: play vs Expert, query SARSA as oracle at each step,
             collect (feature, oracle_action) on states the policy encounters,
             reward-filter (only winning hands), periodically fine-tune.

DAgger fixes the distribution-shift problem of pure BC:
  - BC trains on SARSA's play distribution
  - The policy encounters DIFFERENT states during play
  - DAgger labels the policy's states with SARSA's action → retrains
  - This is proven to converge to near-teacher performance

Combined with reward filtering (only clone winning-hand actions),
this should PUSH PAST pure SARSA's 47.4% WR.

Usage:
    python -u train/train_bnn_policy.py [num_hands] [output_path] [distill_hands] [sarsa_model]

Example:
    python -u train/train_bnn_policy.py 50000 train/results/policy/bnn_policy.pt 20000 train/sarsa_final.pkl

Skip Phase 1 (resume from pretrained):
    python -u train/train_bnn_policy.py 50000 train/results/policy/bnn_policy.pt \\
        --skip_phase1 train/results/policy/bnn_policy_bc_pretrained.pt
"""
from __future__ import annotations

import sys
import os
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from game.engine import GameEngine
from agents.nn_mc_agent import (
    BNN_PolicyNet, BNN_PolicyAgent,
    collect_policy_distill_data, train_bnn_policy_distill,
)
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent


# =========================================================================
#  Phase 1: SARSA Behavioral Cloning - Supervised Pretrain
# =========================================================================

def phase1_distill_and_pretrain(
    distill_hands=20000,
    pretrain_epochs=150,
    batch_size=64,
    lr=5e-4,
    sarsa_model_path="train/sarsa_final.pkl",
    model_save_path="train/results/policy/bnn_policy_pretrained.pt",
    mask_prob=0.5,
):
    print("=" * 60)
    print("  Phase 1: SARSA Behavioral Cloning - Supervised Pretrain")
    print(f"  SARSA model:      {sarsa_model_path}")
    print(f"  Distill hands:    {distill_hands}")
    print(f"  Feature masking:  {mask_prob:.0%}")
    print(f"  Pretrain epochs:  {pretrain_epochs}")
    print(f"  Learning rate:    {lr}")
    print(f"  Loss:             CrossEntropy (classification)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print(f"\n  Loading SARSA teacher from {sarsa_model_path}...")
    sarsa_agent = SarsaAgent(name="SARSA_teacher", epsilon=0.0,
                             load_q_table_path=sarsa_model_path)
    print(f"  SARSA Q-table size: {len(sarsa_agent.q_table)} states")

    print(f"\n  Collecting behavioral cloning data ({distill_hands} hands)...")
    t0 = time.time()
    expert = ExpertAgent(name="Expert")
    env = GameEngine(sarsa_agent, expert)

    X, y, mask_flags = collect_policy_distill_data(
        env, sarsa_agent, num_hands=distill_hands,
        mask_prob=mask_prob, verbose=True, observer_player=0)

    elapsed = time.time() - t0
    print(f"\n  Collected {len(X)} samples in {elapsed:.1f}s "
          f"({len(X)/distill_hands:.1f} samples/hand)")

    for i, name in enumerate(["FOLD", "CALL", "RAISE"]):
        cnt = np.sum(y == i)
        print(f"    {name}: {cnt:>6} ({cnt/len(y)*100:5.1f}%)")
    print(f"  Masked samples: {np.sum(mask_flags)}/{len(X)} = {np.mean(mask_flags):.1%}")

    print(f"\n  Pretraining BNN Policy Network ({pretrain_epochs} epochs)...")
    model = BNN_PolicyNet(input_dim=47).to(device)
    model = train_bnn_policy_distill(
        model, X, y, mask_flags=mask_flags,
        epochs=pretrain_epochs, batch_size=batch_size, lr=lr,
        val_split=0.15, device=device, verbose=True)

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save({
        "policy_net_state_dict": model.state_dict(),
        "distill_samples": len(X),
        "sarsa_model": sarsa_model_path,
    }, model_save_path)
    print(f"\n  Pretrained policy network saved to {model_save_path}")
    return model, device


# =========================================================================
#  Phase 2: DAgger — Online Policy Distillation with SARSA as Oracle
# =========================================================================
#
#  At each agent decision point during play:
#    1. Agent selects action (epsilon-greedy from network)
#    2. SARSA oracle provides greedy action as ground truth
#    3. (features, oracle_action) added to DAgger buffer
#    4. Periodically fine-tune on DAgger buffer
#
#  Reward filter: only add samples from hands that yielded positive return.
#  This de-emphasizes losing patterns and reinforces winning ones.

def train_one_hand_dagger(env, agent, sarsa_agent, agent_id=0):
    """Play one hand with DAgger data collection."""
    obs = env.reset_hand()
    agent.reset()
    done = False
    step_count = 0
    agent_reward = 0.0

    hand_samples = []  # (features, oracle_action) for this hand

    while not done:
        step_count += 1
        if step_count > 50:
            break

        cp = env.current_player

        if cp == agent_id:
            features = agent._feat_builder._encode_bnn_features(obs)
            action = agent.act(obs)

            # SARSA oracle: what would SARSA do?
            sarsa_state = sarsa_agent._encode_state(obs)
            q_vals = sarsa_agent.q_table[sarsa_state]
            legal = obs.legal_actions
            best_val = max(q_vals[a] for a in legal)
            best_actions = [a for a in legal if q_vals[a] == best_val]
            oracle_action = random.choice(best_actions) if len(best_actions) > 1 else best_actions[0]

            hand_samples.append((features, oracle_action))

            round_before = obs.current_round
            obs, reward, done, info = env.step(action)
            agent.record_action(cp, action, round_before)

            if done:
                agent_reward = info.get("result").rewards[agent_id]
        else:
            round_before = obs.current_round
            opp_action = env.agents[cp].act(obs)
            obs, reward, done, info = env.step(opp_action)
            agent.record_action(cp, opp_action, round_before)

            if done:
                agent_reward = info.get("result").rewards[agent_id]

    # Reward filter: only add samples from winning hands
    if agent_reward > 0:
        for feat, oracle_act in hand_samples:
            agent.add_dagger_sample(feat, oracle_act)

    agent.decay_epsilon()
    return agent_reward


def phase2_dagger(
    pretrained_model, device, num_hands=50000,
    sarsa_model_path="train/sarsa_final.pkl",
    output_path="train/results/policy/bnn_policy_final.pt",
    dagger_lr=1e-4,
    dagger_interval=2000,
    dagger_epochs=10,
    dagger_capacity=50000,
):
    print("\n" + "=" * 60)
    print("  Phase 2: DAgger — Online Policy Distillation")
    print("=" * 60)
    print(f"  Total hands:         {num_hands}")
    print(f"  epsilon schedule:    1.0 -> 0.05 (decay=0.9995)")
    print(f"  DAgger LR:           {dagger_lr}")
    print(f"  Fine-tune interval:  every {dagger_interval} hands")
    print(f"  Fine-tune epochs:    {dagger_epochs}")
    print(f"  Buffer capacity:     {dagger_capacity}")
    print(f"  Reward filter:       only winning hands (return > 0)")
    print(f"  Oracle:              SARSA (epsilon=0, Q-table lookup)")
    print(f"  Output:              {output_path}")
    print("=" * 60)

    # Load SARSA oracle
    print(f"\n  Loading SARSA oracle from {sarsa_model_path}...")
    sarsa_agent = SarsaAgent(name="SARSA_oracle", epsilon=0.0,
                             load_q_table_path=sarsa_model_path)

    # Init policy agent with DAgger
    agent = BNN_PolicyAgent(
        name="BNN_Policy_DAgger",
        epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05,
        mc_samples=20, device=device,
    )
    agent.policy_net.load_state_dict(pretrained_model.state_dict())
    agent.init_dagger(lr=dagger_lr, capacity=dagger_capacity)

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000

    for hand in range(1, num_hands + 1):
        r = train_one_hand_dagger(env, agent, sarsa_agent, agent_id=0)
        chips_window += r
        if r > 0:
            wins_window += 1

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_chips = chips_window / window_size
            wr = wins_window / window_size
            buf_size = len(agent.dagger_buffer)
            print(
                f"| Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                f"AvgChips={avg_chips:+.1f} | "
                f"WR={wr:.1%} | dagger={buf_size} | time={elapsed:.1f}s"
            )
            chips_window = 0.0
            wins_window = 0

        # Periodic DAgger fine-tuning
        if hand % dagger_interval == 0 and len(agent.dagger_buffer) >= 500:
            d_loss, d_acc = agent.train_dagger(epochs=dagger_epochs, batch_size=128)
            print(f"  [DAgger @ {hand}] buffer={len(agent.dagger_buffer)} "
                  f"loss={d_loss:.4f} acc={d_acc:.3f}")

    agent.save_model(output_path)
    total_time = time.time() - start
    print(f"\nPhase 2 completed in {total_time:.1f}s ({total_time / 60:.1f}min).")
    print(f"Final DAgger buffer: {len(agent.dagger_buffer)} samples")


# =========================================================================
#  Main
# =========================================================================

def main():
    # Parse args, handling --skip_phase1 before positional args
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    skip_phase1 = "--skip_phase1" in flags

    num_hands = int(args[0]) if len(args) > 0 else 50000
    output_path = args[1] if len(args) > 1 else "train/results/policy/bnn_policy.pt"
    distill_hands = int(args[2]) if len(args) > 2 else 20000
    sarsa_model = args[3] if len(args) > 3 else "train/sarsa_final.pkl"
    pretrain_path = args[4] if len(args) > 4 else "train/results/policy/bnn_policy_bc_pretrained.pt"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if skip_phase1 and os.path.exists(pretrain_path):
        print("=" * 60)
        print("  RESUME MODE: Skipping Phase 1, loading pretrained policy")
        print(f"  Model: {pretrain_path}")
        print("=" * 60)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BNN_PolicyNet(input_dim=47).to(device)
        checkpoint = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(checkpoint["policy_net_state_dict"])
        print(f"  Loaded (distill_samples={checkpoint.get('distill_samples', '?')})")
        phase2_dagger(
            model, device, num_hands=num_hands,
            sarsa_model_path=sarsa_model,
            output_path=output_path,
            dagger_lr=1e-4,
            dagger_interval=2000,
            dagger_epochs=10,
        )
    else:
        pretrained_model, device = phase1_distill_and_pretrain(
            distill_hands=distill_hands,
            pretrain_epochs=150,
            sarsa_model_path=sarsa_model,
            model_save_path=pretrain_path,
            mask_prob=0.5,
        )
        phase2_dagger(
            pretrained_model, device, num_hands=num_hands,
            sarsa_model_path=sarsa_model,
            output_path=output_path,
            dagger_lr=1e-4,
            dagger_interval=2000,
            dagger_epochs=10,
        )


if __name__ == "__main__":
    main()
