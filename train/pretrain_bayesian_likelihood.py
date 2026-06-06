# train/pretrain_bayesian_likelihood.py
"""
Pretrain Bayesian likelihood table for BayesianAgent.

This script collects empirical counts for:

    P(action | true_hand_strength_bucket, betting_level)

by running games between existing agents and observing the target player's
actions together with their true cards.

Usage:
    python train/pretrain_bayesian_likelihood.py [num_hands] [output_path] [target_agent] [other_agent]

Examples:
    # Quick test
    python train/pretrain_bayesian_likelihood.py 200 train/bayesian_likelihood.pkl expert random

    # More data from Expert behavior
    python train/pretrain_bayesian_likelihood.py 5000 train/bayesian_likelihood.pkl expert random

    # Bluff-rich data
    python train/pretrain_bayesian_likelihood.py 5000 train/bayesian_likelihood_aggressive.pkl aggressive random
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

# Allow running from project root:
# python train/pretrain_bayesian_likelihood.py ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import GameEngine
from game.constants import ACTION_NAMES, ROUND_NAMES
from game.evaluator import compute_equity

from agents.random_agent import RandomAgent
from agents.expert_agent import ExpertAgent
from agents.aggressive_agent import AggressiveAgent, TightPassiveAgent

from agents.bayesian_model import (
    BayesianLikelihoodEstimator,
    equity_to_strength_bucket,
    strength_bucket_name,
)


def create_agent(agent_type: str, player_id: int):
    """
    Create one of the existing agents for data collection.

    Supported:
        random
        expert
        aggressive
        tight_passive
    """
    agent_type = agent_type.lower()

    if agent_type == "random":
        return RandomAgent(name=f"Random_p{player_id}")

    if agent_type == "expert":
        return ExpertAgent(name=f"Expert_p{player_id}")

    if agent_type == "aggressive":
        return AggressiveAgent(name=f"Aggressive_p{player_id}")

    if agent_type in ("tight", "tight_passive", "tightpassive"):
        return TightPassiveAgent(name=f"TightPassive_p{player_id}")

    raise ValueError(
        f"Unknown agent_type={agent_type}. "
        f"Supported: random, expert, aggressive, tight_passive"
    )


def compute_true_bucket(engine: GameEngine, player: int, sim: int = 50) -> int:
    """
    Compute the true strength bucket of a player using actual private cards.

    During offline likelihood pretraining, we are allowed to inspect
    the target player's hole cards.

    Args:
        engine: GameEngine
        player: player id
        sim: Monte Carlo samples for equity computation

    Returns:
        0 weak, 1 mid, 2 strong
    """
    hole = engine.players[player].hole_cards
    community = engine.community_cards

    if len(hole) != 2:
        # Should not happen after reset_hand, but keep robust fallback.
        return 1

    equity = compute_equity(hole, community, sim=sim)
    return equity_to_strength_bucket(equity)


def collect_likelihood(
    num_hands: int,
    target_agent_type: str = "expert",
    other_agent_type: str = "random",
    target_player_id: int = 1,
    equity_sim: int = 50,
    verbose: bool = True,
) -> BayesianLikelihoodEstimator:
    """
    Run games and collect likelihood counts from target player's actions.

    Default setup:
        player 0 = other_agent
        player 1 = target_agent

    We collect only target_player_id's actions.

    Args:
        num_hands: number of hands to simulate
        target_agent_type: behavior we want to model
        other_agent_type: opponent used to generate situations
        target_player_id: usually 1
        equity_sim: MC samples for true equity label
        verbose: print progress

    Returns:
        BayesianLikelihoodEstimator
    """
    if target_player_id not in (0, 1):
        raise ValueError("target_player_id must be 0 or 1.")

    if target_player_id == 0:
        agent0 = create_agent(target_agent_type, 0)
        agent1 = create_agent(other_agent_type, 1)
    else:
        agent0 = create_agent(other_agent_type, 0)
        agent1 = create_agent(target_agent_type, 1)

    env = GameEngine(agent0, agent1)
    estimator = BayesianLikelihoodEstimator(laplace=1.0)

    # Debug/statistics counters
    action_counts = defaultdict(int)
    bucket_counts = defaultdict(int)
    round_counts = defaultdict(int)
    bet_level_counts = defaultdict(int)

    start = time.time()

    for hand in range(1, num_hands + 1):
        obs = env.reset_hand()

        # Reset agents if they maintain per-hand state
        for a in env.agents:
            if hasattr(a, "reset"):
                a.reset()

        done = False
        step_count = 0

        while not done:
            step_count += 1
            if step_count > 80:
                # Safety guard against unexpected infinite loops.
                break

            cp = env.current_player
            acting_agent = env.agents[cp]

            # Observation is already for current player.
            action = acting_agent.act(obs)

            # Record target player's action BEFORE env.step(action),
            # because obs describes the state at decision time.
            if cp == target_player_id:
                true_bucket = compute_true_bucket(
                    env,
                    player=cp,
                    sim=equity_sim,
                )
                bet_level = obs.betting_level

                estimator.record(
                    hand_bucket=true_bucket,
                    bet_level=bet_level,
                    action=action,
                )

                action_counts[action] += 1
                bucket_counts[true_bucket] += 1
                round_counts[obs.current_round] += 1
                bet_level_counts[bet_level] += 1

            obs, reward, done, info = env.step(action)

        if verbose and hand % max(1, num_hands // 10) == 0:
            elapsed = time.time() - start
            print(
                f"  {hand:>7}/{num_hands} hands "
                f"({hand / num_hands:>5.1%}) | "
                f"records={estimator.total_records():>7} | "
                f"time={elapsed:.1f}s"
            )

        # Reset bankrupt players to keep simulation going.
        for i in range(2):
            if env.players[i].chips <= 0:
                env.players[i].chips = 1000

    if verbose:
        print()
        print("=" * 60)
        print("Collection summary")
        print("=" * 60)
        print(f"Target agent:      {target_agent_type}")
        print(f"Other agent:       {other_agent_type}")
        print(f"Target player id:  {target_player_id}")
        print(f"Hands:             {num_hands}")
        print(f"Total records:     {estimator.total_records()}")
        print()

        print("Action counts:")
        for a in sorted(action_counts):
            print(f"  {ACTION_NAMES[a]:<5}: {action_counts[a]}")

        print()
        print("Strength bucket counts:")
        for b in sorted(bucket_counts):
            print(f"  {strength_bucket_name(b):<6}: {bucket_counts[b]}")

        print()
        print("Round counts:")
        for r in sorted(round_counts):
            print(f"  {ROUND_NAMES.get(r, r):<8}: {round_counts[r]}")

        print()
        print("Bet level counts:")
        for b in sorted(bet_level_counts):
            print(f"  B={b}: {bet_level_counts[b]}")

        print()
        print(estimator.summary())

    return estimator


def main():
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "train/bayesian_likelihood.pkl"
    target_agent_type = sys.argv[3] if len(sys.argv) > 3 else "expert"
    other_agent_type = sys.argv[4] if len(sys.argv) > 4 else "random"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("=" * 60)
    print("  Bayesian Likelihood Pretraining")
    print("=" * 60)
    print(f"  num_hands:    {num_hands}")
    print(f"  output_path:  {output_path}")
    print(f"  target_agent: {target_agent_type}")
    print(f"  other_agent:  {other_agent_type}")
    print("=" * 60)

    estimator = collect_likelihood(
        num_hands=num_hands,
        target_agent_type=target_agent_type,
        other_agent_type=other_agent_type,
        target_player_id=1,
        equity_sim=50,
        verbose=True,
    )

    estimator.save(output_path)
    print(f"\nSaved Bayesian likelihood to: {output_path}")


if __name__ == "__main__":
    main()