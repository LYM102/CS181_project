# train/train_bayesian.py
"""
Train BayesianAgent with SARSA-style Q-table updates.

Usage:
    python train/train_bayesian.py [num_hands] [output_path] [likelihood_path] [opponent_type] [reward_mode] [eval_interval] [eval_hands]

reward_mode:
    chips:
        Original chip reward.

    win_loss:
        Win-rate optimized reward:
            win  -> +1
            tie  -> 0
            lose -> -1

Examples:
    # 1000-hand smoke test, evaluate every 1000 hands with 1000 eval hands
    python train/train_bayesian.py 1000 train/bayesian_agent_expert1k.pkl train/bayesian_likelihood.pkl expert win_loss 1000 1000

    # 100k training, evaluate every 10k hands with 1000 eval hands
    python train/train_bayesian.py 100000 train/bayesian_agent_expert100k.pkl train/bayesian_likelihood.pkl expert win_loss 10000 1000
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import GameEngine
from game.constants import STARTING_CHIPS

from agents.bayesian_agent import BayesianAgent
from agents.random_agent import RandomAgent
from agents.expert_agent import ExpertAgent
from agents.aggressive_agent import AggressiveAgent, TightPassiveAgent


def create_opponent(opponent_type: str, player_id: int = 1):
    """
    Create opponent agent by name.
    """
    opponent_type = opponent_type.lower()

    if opponent_type == "random":
        return RandomAgent(name=f"Random_p{player_id}")

    if opponent_type == "expert":
        return ExpertAgent(name=f"Expert_p{player_id}")

    if opponent_type == "aggressive":
        return AggressiveAgent(name=f"Aggressive_p{player_id}")

    if opponent_type in ("tight", "tight_passive", "tightpassive"):
        return TightPassiveAgent(name=f"TightPassive_p{player_id}")

    raise ValueError(
        f"Unknown opponent_type={opponent_type}. "
        f"Supported: random, expert, aggressive, tight_passive"
    )


# ============================================================
# Reward helpers
# ============================================================

def get_winner(result):
    """
    Robustly get winner from HandResult.

    Return:
        0 / 1 if someone wins,
        None if tie or invalid.
    """
    if result is None:
        return None

    winner = getattr(result, "winner", None)

    if winner in (0, 1):
        return winner

    return None


def is_agent_winner(result, agent_id: int) -> bool:
    return get_winner(result) == agent_id


def compute_training_reward(result, agent_id: int, reward_mode: str) -> float:
    """
    Compute terminal training reward.
    """
    reward_mode = reward_mode.lower()

    if result is None:
        return 0.0

    if reward_mode == "chips":
        return float(result.rewards[agent_id])

    if reward_mode in ("win_loss", "winrate", "win"):
        winner = get_winner(result)

        if winner == agent_id:
            return 1.0

        if winner is None:
            return 0.0

        return -1.0

    raise ValueError(
        f"Unknown reward_mode={reward_mode}. "
        f"Supported: chips, win_loss"
    )


def save_eval_model(agent: BayesianAgent, path: str) -> None:
    """
    Save a copy of the current model with epsilon=0.

    Use this model in main.py evaluation.
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    agent.save_model(path)
    agent.epsilon = old_epsilon


# ============================================================
# Training one hand
# ============================================================

def train_one_hand(
    env: GameEngine,
    agent: BayesianAgent,
    agent_id: int = 0,
    reward_mode: str = "win_loss",
) -> tuple[float, bool]:
    """
    Train one hand.

    Returns:
        train_reward:
            reward used for Q-learning.

        won:
            whether BayesianAgent won this hand.
    """
    obs = env.reset_hand()
    agent.reset()

    done = False
    step_count = 0

    prev_state = None
    prev_action = None

    agent_reward = 0.0
    agent_won = False

    while not done:
        step_count += 1
        if step_count > 80:
            break

        cp = env.current_player

        if cp == agent_id:
            # BayesianAgent turn.
            state = agent._encode_state(obs)
            action = agent.act(obs)

            if prev_state is not None:
                agent.learn_sarsa(
                    prev_state,
                    prev_action,
                    0.0,
                    state,
                    action,
                    done=False,
                )

            obs, reward, done, info = env.step(action)

            if done:
                result = info.get("result")
                agent_reward = compute_training_reward(
                    result,
                    agent_id=agent_id,
                    reward_mode=reward_mode,
                )
                agent_won = is_agent_winner(result, agent_id)

                agent.learn_sarsa(
                    state,
                    action,
                    agent_reward,
                    None,
                    None,
                    done=True,
                )
            else:
                prev_state = state
                prev_action = action

        else:
            # Opponent turn.
            bet_level_before = obs.betting_level
            round_before = obs.current_round

            opp_action = env.agents[cp].act(obs)

            agent.record_action(
                player=cp,
                action=opp_action,
                round_num=round_before,
                bet_level=bet_level_before,
            )

            obs, reward, done, info = env.step(opp_action)

            if done:
                result = info.get("result")
                agent_reward = compute_training_reward(
                    result,
                    agent_id=agent_id,
                    reward_mode=reward_mode,
                )
                agent_won = is_agent_winner(result, agent_id)

                if prev_state is not None:
                    agent.learn_sarsa(
                        prev_state,
                        prev_action,
                        agent_reward,
                        None,
                        None,
                        done=True,
                    )

    agent.decay_epsilon()

    for i in range(2):
        if env.players[i].chips <= 0:
            env.players[i].chips = STARTING_CHIPS

    return agent_reward, agent_won


# ============================================================
# Evaluation inside training script
# ============================================================

def evaluate_agent(
    agent: BayesianAgent,
    opponent,
    num_hands: int = 1000,
    agent_id: int = 0,
) -> tuple[float, float]:
    """
    Evaluate BayesianAgent with epsilon=0.

    Returns:
        avg_chip_reward, win_rate
    """
    env = GameEngine(agent, opponent)

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total_reward = 0.0
    wins = 0

    for _ in range(num_hands):
        obs = env.reset_hand()
        agent.reset()

        done = False
        step_count = 0

        while not done:
            step_count += 1
            if step_count > 80:
                break

            cp = env.current_player

            if cp == agent_id:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
            else:
                bet_level_before = obs.betting_level
                round_before = obs.current_round

                opp_action = env.agents[cp].act(obs)

                agent.record_action(
                    player=cp,
                    action=opp_action,
                    round_num=round_before,
                    bet_level=bet_level_before,
                )

                obs, reward, done, info = env.step(opp_action)

        if done:
            result = info.get("result")
            total_reward += result.rewards[agent_id]

            if is_agent_winner(result, agent_id):
                wins += 1

        for i in range(2):
            if env.players[i].chips <= 0:
                env.players[i].chips = STARTING_CHIPS

    agent.epsilon = old_epsilon

    return total_reward / num_hands, wins / num_hands


# ============================================================
# Main
# ============================================================

def main():
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "train/bayesian_agent_expert100k.pkl"
    likelihood_path = sys.argv[3] if len(sys.argv) > 3 else "train/bayesian_likelihood.pkl"
    opponent_type = sys.argv[4] if len(sys.argv) > 4 else "expert"
    reward_mode = sys.argv[5] if len(sys.argv) > 5 else "win_loss"
    eval_interval = int(sys.argv[6]) if len(sys.argv) > 6 else 10000
    eval_hands = int(sys.argv[7]) if len(sys.argv) > 7 else 1000

    reward_mode = reward_mode.lower()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".pkl"

    best_eval_path = f"{base}_best_eval{ext}"
    final_eval_path = f"{base}_final_eval{ext}"

    print("=" * 60)
    print("  BayesianAgent Training: Expert Win-Rate Mode")
    print("=" * 60)
    print(f"  num_hands:       {num_hands}")
    print(f"  output_path:     {output_path}")
    print(f"  best_eval_path:  {best_eval_path}")
    print(f"  final_eval_path: {final_eval_path}")
    print(f"  likelihood_path: {likelihood_path}")
    print(f"  opponent_type:   {opponent_type}")
    print(f"  reward_mode:     {reward_mode}")
    print(f"  eval_interval:   {eval_interval}")
    print(f"  eval_hands:      {eval_hands}")
    print("=" * 60)

    agent = BayesianAgent(
        name="Bayesian_p0",
        epsilon=1.0,
        epsilon_decay=0.9998,
        epsilon_min=0.02,
        alpha=0.05,
        gamma=1.0,
        player_id=0,
        likelihood_path=likelihood_path,
        explore_avoid_fold=True,
        use_heuristic_fallback=True,
        winrate_no_fold=True,
    )

    opponent = create_opponent(opponent_type, player_id=1)
    env = GameEngine(agent, opponent)

    start = time.time()

    window_size = 1000

    reward_window = 0.0
    wins_window = 0

    best_eval_wr = -1.0
    best_eval_avg_chips = 0.0
    best_eval_hand = 0

    for hand in range(1, num_hands + 1):
        r, won = train_one_hand(
            env,
            agent,
            agent_id=0,
            reward_mode=reward_mode,
        )

        reward_window += r
        if won:
            wins_window += 1

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_train_reward = reward_window / window_size
            train_wr = wins_window / window_size

            print(
                f"| Hand {hand:>7} | "
                f"eps={agent.epsilon:.4f} | "
                f"Qsize={agent.get_q_table_size():>5} | "
                f"TrainReward={avg_train_reward:+.3f} | "
                f"TrainWR={train_wr:.1%} | "
                f"time={elapsed:.1f}s"
            )

            reward_window = 0.0
            wins_window = 0

        if eval_interval > 0 and hand % eval_interval == 0:
            eval_opp = create_opponent(opponent_type, player_id=1)

            avg_chips, eval_wr = evaluate_agent(
                agent,
                eval_opp,
                num_hands=eval_hands,
                agent_id=0,
            )

            print(
                f">>> EVAL {hand // 1000}k | "
                f"WR vs {opponent_type}: {eval_wr * 100:.1f}% | "
                f"AvgChips={avg_chips:+.2f}"
            )

            # Save latest training checkpoint.
            agent.save_model(output_path)

            # Save best eval checkpoint with epsilon=0.
            if eval_wr > best_eval_wr:
                best_eval_wr = eval_wr
                best_eval_avg_chips = avg_chips
                best_eval_hand = hand

                save_eval_model(agent, best_eval_path)

                print(
                    f">>> NEW BEST | "
                    f"hand={best_eval_hand} | "
                    f"WR={best_eval_wr * 100:.1f}% | "
                    f"AvgChips={best_eval_avg_chips:+.2f} | "
                    f"saved={best_eval_path}"
                )

    # Save final training model and final eval model.
    agent.save_model(output_path)
    save_eval_model(agent, final_eval_path)

    total_time = time.time() - start

    print()
    print("=" * 60)
    print("Training completed")
    print("=" * 60)
    print(f"Total time:       {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Final Q-table:    {agent.get_q_table_size()}")
    print(f"Saved train:      {output_path}")
    print(f"Saved best eval:  {best_eval_path}")
    print(f"Saved final eval: {final_eval_path}")

    if best_eval_wr >= 0:
        print()
        print("Best checkpoint during training:")
        print(f"  hand:      {best_eval_hand}")
        print(f"  WR:        {best_eval_wr * 100:.1f}%")
        print(f"  AvgChips:  {best_eval_avg_chips:+.2f}")
        print(f"  model:     {best_eval_path}")

    print()
    print("Final evaluation inside training script:")
    final_opp = create_opponent(opponent_type, player_id=1)
    avg_chips, wr = evaluate_agent(
        agent,
        final_opp,
        num_hands=eval_hands,
        agent_id=0,
    )
    print(
        f"  Bayesian vs {opponent_type:<6}: "
        f"WR={wr * 100:.1f}% | AvgChips={avg_chips:+.2f}"
    )


if __name__ == "__main__":
    main()