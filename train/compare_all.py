# train/compare_all.py — Train SARSA, then compare Our model vs SARSA vs Expert
"""
Usage:
    python -u train/compare_all.py
"""
from __future__ import annotations
import sys, os, time
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from game.engine import GameEngine
from agents.nn_mc_agent import BNN_PolicyNet, BNN_PolicyAgent
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent


SARSA_MODEL = "train/results/policy/sarsa_trained.pkl"
OUR_MODEL = "train/results/policy/expert_distill_sp_sp.pt"


# =========================================================================
#  Evaluation helper (win rate + average reward)
# =========================================================================
def evaluate(agent0, agent1, name0, name1, num_hands=3000, agent0_id=0):
    env = GameEngine(agent0, agent1) if agent0_id == 0 else GameEngine(agent1, agent0)
    wins = [0, 0]
    ties = 0
    net_rewards = [0.0, 0.0]  # true zero-sum: chip change per player
    for a in env.agents:
        a.reset()
    for _ in range(num_hands):
        for a in env.agents:
            a.reset()
        # Record chips before the hand (after reset so players are initialized)
        obs = env.reset_hand()
        chips_before = [env.players[i].chips for i in range(2)]
        done = False
        while not done:
            cp = env.current_player
            action = env.agents[cp].act(obs)
            round_before = obs.current_round
            obs, reward, done, info = env.step(action)
            for pid in range(2):
                if hasattr(env.agents[pid], 'record_action'):
                    env.agents[pid].record_action(cp, action, round_before)
        result = info.get("result")
        if result:
            if result.winner is not None:
                wins[result.winner] += 1
            else:
                ties += 1
            # True zero-sum: net chip change from engine state
            for pid in range(2):
                net_rewards[pid] += env.players[pid].chips - chips_before[pid]

    wr0 = wins[0] / num_hands
    wr1 = wins[1] / num_hands
    avg_r0 = net_rewards[0] / num_hands
    avg_r1 = net_rewards[1] / num_hands

    print(f"\n{'='*60}")
    print(f"  {name0} vs {name1} ({num_hands} hands)")
    print(f"{'='*60}")
    print(f"  {name0}: WR={wr0:.1%} ({wins[0]}W/{wins[1]}L/{ties}T)  AvgR={avg_r0:+.2f}")
    print(f"  {name1}: WR={wr1:.1%} ({wins[1]}W/{wins[0]}L/{ties}T)  AvgR={avg_r1:+.2f}")

    # Return metrics for agent0 side
    return wr0, avg_r0


# =========================================================================
#  Load our best model
# =========================================================================
def load_our_model():
    print(f"\n  Loading our model from {OUR_MODEL}...")
    checkpoint = torch.load(OUR_MODEL, map_location="cpu")
    arch_cfg = checkpoint.get("arch_config", {})
    hidden_dims = arch_cfg.get("hidden_dims", (256, 128, 64))
    dropout_rate = arch_cfg.get("dropout_rate", 0.2)
    use_residual = arch_cfg.get("use_residual", True)
    use_layernorm = arch_cfg.get("use_layernorm", True)
    agent = BNN_PolicyAgent(
        name="Our_BNN_SP",
        epsilon=0.0, device="cpu",
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_residual=use_residual,
        use_layernorm=use_layernorm,
    )
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    return agent


# =========================================================================
#  Train SARSA
# =========================================================================
def train_sarsa():
    if os.path.exists(SARSA_MODEL):
        print(f"  SARSA model found at {SARSA_MODEL}, loading...")
        return SarsaAgent(name="SARSA", load_q_table_path=SARSA_MODEL)

    print("\n" + "=" * 60)
    print("  Training SARSA vs Expert (100,000 hands)")
    print("=" * 60)

    sarsa = SarsaAgent(name="SARSA", epsilon=1.0, alpha=0.05, gamma=0.95)
    expert = ExpertAgent(name="Expert")
    env = GameEngine(sarsa, expert)

    start = time.time()
    for hand in range(1, 100001):
        sarsa.reset()
        expert.reset()
        obs = env.reset_hand()
        done = False
        prev_state = None
        prev_action = None

        while not done:
            cp = env.current_player
            if cp == 0:
                action = sarsa.act(obs)
                state = sarsa._encode_state(obs)
                obs, reward, done, info = env.step(action)
                if prev_state is not None:
                    # Update previous SARSA transition (intermediate)
                    sarsa.learn(prev_state, prev_action,
                                0.0, state, action, done=False)
                prev_state = state
                prev_action = action
                if done:
                    # SARSA's own action ended the hand — terminal update for THIS action
                    final_reward = info.get("result").rewards[0]
                    sarsa.learn(state, action, final_reward,
                                None, None, done=True)
            else:
                action = expert.act(obs)
                obs, reward, done, info = env.step(action)
                if done and prev_state is not None:
                    # Expert's action ended hand — terminal update for SARSA's last action
                    final_reward = info.get("result").rewards[0]
                    sarsa.learn(prev_state, prev_action,
                                final_reward, None, None, done=True)

        sarsa.decay_epsilon()

        if hand % 10000 == 0:
            elapsed = time.time() - start
            print(f"  Hand {hand:>6} / 100000  (epsilon={sarsa.epsilon:.4f}, Q-size={sarsa.get_q_table_size()}, {elapsed:.0f}s)")

    total_time = time.time() - start
    print(f"\n  SARSA training completed in {total_time:.0f}s ({total_time/60:.1f}min)")

    os.makedirs(os.path.dirname(SARSA_MODEL), exist_ok=True)
    sarsa.save_q_table(SARSA_MODEL)
    print(f"  SARSA model saved to {SARSA_MODEL}")
    return sarsa


# =========================================================================
#  Main
# =========================================================================
def main():
    print("=" * 60)
    print("  Three-way Comparison: BNN-SP2 vs SARSA vs Expert")
    print("=" * 60)

    # 1. Train/Load SARSA
    sarsa = train_sarsa()
    sarsa.epsilon = 0.0

    # 2. Load our best model
    our = load_our_model()

    # 3. Run comparisons
    print("\n\n" + "=" * 60)
    print("  RUNNING COMPARISONS (3000 hands each)")
    print("=" * 60)

    # a) Our model vs Expert
    our_wr_expert, our_ar_expert = evaluate(
        our, ExpertAgent(name="Expert"), "Our_BNN_SP2", "Expert", num_hands=3000, agent0_id=0)

    # b) Our model vs SARSA
    our_wr_sarsa, our_ar_sarsa = evaluate(
        our, sarsa, "Our_BNN_SP2", "SARSA", num_hands=3000, agent0_id=0)

    # c) SARSA vs Expert
    sarsa_wr_expert, sarsa_ar_expert = evaluate(
        sarsa, ExpertAgent(name="Expert"), "SARSA", "Expert", num_hands=3000, agent0_id=0)

    # 4. Summary table
    print("\n\n" + "=" * 60)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Matchup':<30} {'WR':>8} {'AvgR':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Our BNN-SP2 vs Expert':<30} {our_wr_expert:>7.1%} {our_ar_expert:>+9.2f}")
    print(f"  {'Our BNN-SP2 vs SARSA':<30} {our_wr_sarsa:>7.1%} {our_ar_sarsa:>+9.2f}")
    print(f"  {'SARSA vs Expert':<30} {sarsa_wr_expert:>7.1%} {sarsa_ar_expert:>+9.2f}")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()
