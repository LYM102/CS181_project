# train/train_sarsa.py — SARSA On-Policy TD Training for 52-card Texas Hold'em
"""
SARSA (State-Action-Reward-State-Action) training against ExpertAgent (CFR).

Key design decisions:
  - Single-stage: train directly against ExpertAgent (no random warm-up phase).
    Rationale: random opponent teaches bad habits; Expert provides meaningful gradients.
  - Agent/opponent turn separation: agent only learns on its own turns via SARSA TD updates.
    Opponent's actions provide transition and terminal reward but no learning signal.
  - ε-greedy with slow decay: ε 1.0 → 0.10 (0.9998 per hand), ~12k hands before ε < 0.15.

Usage:
    conda activate fmd
    python train/train_sarsa.py [num_hands] [output_path]

Examples:
    # Quick test
    python train/train_sarsa.py 100000 train/sarsa_vs_expert.pkl

    # Full training (slurm)
    python -u train/train_sarsa.py 200000 train/results/XXXX/sarsa_vs_expert.pkl
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import GameEngine
from agents.sarsa_agent import SarsaAgent
from agents.expert_agent import ExpertAgent
from agents.random_agent import RandomAgent


def train_one_hand(env: GameEngine, agent: SarsaAgent, agent_id: int = 0):
    """
    Train one hand: agent acts on own turns with SARSA TD updates.
    Opponent's turns only advance the game state (no agent learning).

    Returns:
        net reward for agent_id
    """
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
            # ===== Agent's turn =====
            state = agent._encode_state(obs)
            action = agent.act(obs)

            # SARSA update on previous (s, a) — intermediate reward = 0
            if prev_state is not None:
                agent.learn(
                    prev_state, prev_action, 0.0,
                    state, action, done=False,
                )

            obs, reward, done, info = env.step(action)

            if done:
                agent_reward = info.get("result").rewards[agent_id]
                agent.learn(
                    state, action, agent_reward,
                    None, None, done=True,
                )
            else:
                prev_state, prev_action = state, action
        else:
            # ===== Opponent's turn =====
            opp_action = env.agents[cp].act(obs)
            obs, reward, done, info = env.step(opp_action)

            if done and prev_state is not None:
                # Opponent ended hand — terminal update for agent's last (s,a)
                agent_reward = info.get("result").rewards[agent_id]
                agent.learn(
                    prev_state, prev_action, agent_reward,
                    None, None, done=True,
                )

    agent.decay_epsilon()
    return agent_reward


def evaluate_agent(agent: SarsaAgent, opponent, num_hands: int = 500):
    """Evaluate agent (epsilon=0, no training) — returns average reward."""
    env = GameEngine(agent, opponent)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    total_reward = 0.0
    wins = 0

    for _ in range(num_hands):
        obs = env.reset_hand()
        done = False
        while not done:
            cp = env.current_player
            action = env.agents[cp].act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                total_reward += info.get("result").rewards[0]
                if info.get("result").winner == 0:
                    wins += 1

    agent.epsilon = original_epsilon
    return total_reward / num_hands, wins / num_hands


def main():
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "train/sarsa_vs_expert.pkl"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("=" * 60)
    print("  SARSA Training (52-card, vs ExpertAgent)")
    print(f"  Total hands: {num_hands}")
    print(f"  Output:      {output_path}")
    print("=" * 60)

    # Create agent
    agent = SarsaAgent(
        name="SARSA",
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9998,
        epsilon_min=0.10,
    )

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000
    eval_interval = 20000

    for hand in range(1, num_hands + 1):
        r = train_one_hand(env, agent, agent_id=0)
        chips_window += r
        if r > 0:
            wins_window += 1

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_chips = chips_window / window_size
            wr = wins_window / window_size
            print(
                f"| Hand {hand:>7} | ε={agent.epsilon:.4f} | "
                f"Qsize={agent.get_q_table_size():>5} | "
                f"AvgChips={avg_chips:+.1f} | "
                f"WR={wr:.1%} | time={elapsed:.0f}s"
            )
            chips_window = 0.0
            wins_window = 0

        if hand % eval_interval == 0:
            # Evaluate vs Expert
            avg_chips, wr = evaluate_agent(agent, ExpertAgent(), num_hands=500)
            print(f">>> EVAL {hand//1000}k | WR vs Expert: {wr*100:.1f}% | AvgChips: {avg_chips:+.1f}")
            # Save checkpoint
            agent.save_q_table(output_path)

    # Final save
    agent.save_q_table(output_path)
    total_time = time.time() - start
    print(f"\nTraining completed in {total_time:.1f}s ({total_time / 60:.1f}min).")
    print(f"Final Q-table size: {agent.get_q_table_size()}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    for opp_name, opp_cls in [("Expert", ExpertAgent), ("Random", RandomAgent)]:
        avg_chips, wr = evaluate_agent(agent, opp_cls(), num_hands=2000)
        print(f"  SARSA vs {opp_name}: WR={wr*100:.1f}%, AvgChips={avg_chips:+.1f}")


if __name__ == "__main__":
    main()
