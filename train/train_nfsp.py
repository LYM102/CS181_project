# train/train_nfsp.py — NFSP Self-Play Training for 52-card Texas Hold'em
"""
Neural Fictitious Self-Play training via self-play.

Two agents (sharing network parameters) play against each other.
Each uses η-anticipatory dynamics: with prob η use DQN best response,
with prob 1-η use average policy. Over time, average policy → Nash equilibrium.

Key improvements over baseline:
  - γ=0.95 (was 1.0) to prevent DQN value divergence
  - η=0.5 → 0.1 (decay) for more RL samples early, more stable policy late
  - 29-dim state (was 15): opponent action history + board texture
  - Running reward normalization for stable DQN targets
  - Chips-based evaluation metrics (avg_chips/hand, total_chips)

Usage:
    python -u train/train_nfsp.py [num_hands] [output_path]

Example:
    python -u train/train_nfsp.py 1000000 train/results/nfsp_model.pt
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import GameEngine
from game.constants import BETTING_LEVELS
from agents.nfsp_agent import NFSPAgent
from agents.expert_agent import ExpertAgent
from agents.random_agent import RandomAgent
import numpy as np


class SelfPlayWrapper:
    """Wrapper for self-play: two agents sharing network parameters."""

    def __init__(self, shared_agent: NFSPAgent):
        self.shared = shared_agent
        # Create a second agent that shares networks but has separate buffers
        self.agent_copy = NFSPAgent(
            name="NFSPAgent_copy",
            hidden_dim=128,
            eta=shared_agent.eta,
            eta_decay=shared_agent.eta_decay,
            eta_min=shared_agent.eta_min,
            epsilon=shared_agent.epsilon,
            gamma=shared_agent.gamma,
        )
        # Share network parameters (both point to same objects)
        self.agent_copy.q_network = shared_agent.q_network
        self.agent_copy.target_network = shared_agent.target_network
        self.agent_copy.policy_network = shared_agent.policy_network
        self.agent_copy.q_optimizer = shared_agent.q_optimizer
        self.agent_copy.policy_optimizer = shared_agent.policy_optimizer


def train_one_hand_selfplay(engine: GameEngine, agents: list, reward_stats: dict,
                            agent_id: int = 0):
    """
    Train one hand of self-play. Both agents use η-anticipatory dynamics.
    Stores transitions and trains networks after each hand.

    Reward normalization: uses running mean/std of raw chip rewards
    to produce stable DQN targets (replaces the old ÷100 heuristic).

    Returns:
        (normalized_reward, raw_chips, q_loss, p_loss) for agent_id
    """
    obs = engine.reset_hand()

    # Reset opponent tracking for both agents
    for agent in agents:
        if hasattr(agent, 'reset'):
            agent.reset()

    done = False

    # Track states and actions for each player
    trajectories = {0: [], 1: []}  # player -> [(state, action)]

    while not done:
        cp = engine.current_player
        agent = agents[cp]
        opponent = agents[1 - cp]

        state = agent._encode_state(obs)
        action = agent.act_train(obs)

        # Store in SL buffer (always, regardless of mode)
        agent.store_sl_transition(state, action)

        trajectories[cp].append((state, action, agent._current_mode))

        round_before = obs.current_round
        obs, reward, done, info = engine.step(action)

        # Record opponent action for opponent modeling features
        if hasattr(opponent, 'record_opp_action'):
            opponent.record_opp_action(round_before, action)

    # Get rewards
    result = info.get("result")
    if result is None:
        return 0.0, 0.0, 0.0, 0.0

    raw_rewards = {k: float(v) for k, v in result.rewards.items()}
    raw_r = raw_rewards.get(agent_id, 0.0)

    # Update running reward statistics
    reward_stats['count'] += 1
    delta = raw_r - reward_stats['mean']
    reward_stats['mean'] += delta / reward_stats['count']
    reward_stats['M2'] += delta * (raw_r - reward_stats['mean'])
    if reward_stats['count'] >= 2:
        reward_stats['std'] = np.sqrt(reward_stats['M2'] / (reward_stats['count'] - 1))
    else:
        reward_stats['std'] = 100.0  # initial std guess

    # Normalize rewards for DQN training (to ~ N(0,1) scale)
    eps = 1e-8
    safe_std = max(reward_stats['std'], 1.0)  # floor: prevent division by tiny std
    norm_rewards = {pid: r / (safe_std + eps) for pid, r in raw_rewards.items()}

    # Store RL transitions (only for best_response steps)
    for pid in range(2):
        agent = agents[pid]
        traj = trajectories[pid]
        norm_r = norm_rewards.get(pid, 0.0)

        for i, (state, action, mode) in enumerate(traj):
            if mode == 'best_response':
                if i < len(traj) - 1:
                    next_state = traj[i + 1][0]
                    agent.store_rl_transition(state, action, 0.0, next_state, False)
                else:
                    agent.store_rl_transition(state, action, norm_r, state, True)

    # Train networks (shared, so training either agent updates both)
    q_loss = agents[0].train_q_network()
    p_loss = agents[0].train_policy_network()

    norm_r_agent = norm_rewards.get(agent_id, 0.0)
    return norm_r_agent, raw_r, q_loss, p_loss


def evaluate_vs_expert(agent: NFSPAgent, num_hands: int = 1000) -> tuple:
    """
    Evaluate agent (using policy network) vs ExpertAgent.

    Returns:
        (win_rate, avg_chips_per_hand)
    """
    expert = ExpertAgent()
    engine = GameEngine(agent, expert)
    wins = 0
    total_chips = 0.0
    for _ in range(num_hands):
        result = engine.run_hand()
        if result.winner == 0:
            wins += 1
        total_chips += result.rewards.get(0, 0)
    return wins / num_hands, total_chips / num_hands


def train():
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "train/results/nfsp_model.pt"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print(f"  NFSP Self-Play Training (52-card Texas Hold'em)")
    print(f"  Total hands: {num_hands}")
    print(f"  Output:      {output_path}")
    print(f"  Betting:     {BETTING_LEVELS}")
    print(f"  γ=0.95  |  η=0.5→0.1  |  state_dim=29")
    print("=" * 60)

    # Create shared agent with improved hyperparameters
    agent = NFSPAgent(
        name="NFSP_main",
        hidden_dim=128,
        eta=0.5,            # start with more RL exploration (was 0.1)
        eta_decay=0.99999,  # decay per hand: ~0.37× after 100k, ~0.61× after 50k
        eta_min=0.05,
        epsilon=0.06,
        gamma=0.95,         # discount factor (was 1.0 → value divergence)
        q_lr=0.01,
        policy_lr=0.005,
        batch_size=128,
        target_update_freq=1000,
    )

    # Self-play wrapper (shares networks)
    wrapper = SelfPlayWrapper(agent)
    agents = [agent, wrapper.agent_copy]

    # Dummy agents for GameEngine (we manually control the loop)
    engine = GameEngine(RandomAgent("dummy0"), RandomAgent("dummy1"))
    engine.agents = agents  # override with NFSP agents

    # Running reward normalization stats
    reward_stats = {'mean': 0.0, 'M2': 0.0, 'std': 100.0, 'count': 0}

    start = time.time()
    total_reward = 0.0
    total_chips = 0.0
    wins_window = 0
    q_losses = []
    p_losses = []
    eval_interval = 20000
    log_interval = 5000

    for hand in range(1, num_hands + 1):
        norm_r, raw_chips, ql, pl = train_one_hand_selfplay(
            engine, agents, reward_stats, agent_id=0)
        total_reward += norm_r
        total_chips += raw_chips
        if raw_chips > 0:
            wins_window += 1
        if ql > 0:
            q_losses.append(ql)
        if pl > 0:
            p_losses.append(pl)

        # Decay η for both agents after each hand
        agent.decay_eta()
        wrapper.agent_copy.eta = agent.eta  # keep copy in sync

        if hand % log_interval == 0:
            elapsed = time.time() - start
            speed = hand / elapsed
            avg_r_norm = total_reward / log_interval
            avg_chips = total_chips / log_interval
            wr_window = wins_window / log_interval
            avg_ql = np.mean(q_losses[-100:]) if q_losses else 0
            avg_pl = np.mean(p_losses[-100:]) if p_losses else 0
            current_eta = agent.get_eta()
            print(
                f"  {hand//1000}k | Speed: {speed:.0f}/s | "
                f"AvgChips: {avg_chips:+.1f} | WR_self: {wr_window:.1%} | "
                f"R_norm: {avg_r_norm:+.3f}(σ={reward_stats['std']:.0f}) | "
                f"Q_loss: {avg_ql:.3f} | P_loss: {avg_pl:.3f} | "
                f"η={current_eta:.3f} | "
                f"RL_buf: {len(agent.rl_buffer)} | SL_buf: {len(agent.sl_buffer)}",
                flush=True,
            )
            total_reward = 0.0
            total_chips = 0.0
            wins_window = 0

        if hand % eval_interval == 0:
            wr, avg_chips_eval = evaluate_vs_expert(agent, num_hands=1000)
            elapsed = time.time() - start
            print(
                f"  *** EVAL {hand//1000}k | WR vs Expert: {wr*100:.1f}% | "
                f"AvgChips: {avg_chips_eval:+.1f} | "
                f"Time: {elapsed:.0f}s ***",
                flush=True,
            )
            agent.save_model(output_path)

    # Final save
    agent.save_model(output_path)
    total_time = time.time() - start
    print(f"\nTraining completed in {total_time:.1f}s ({total_time / 3600:.2f}h).")

    # Final evaluation
    wr, avg_chips_eval = evaluate_vs_expert(agent, num_hands=2000)
    print(f"Final WR vs Expert: {wr*100:.1f}% | AvgChips: {avg_chips_eval:+.1f}")


if __name__ == "__main__":
    train()
