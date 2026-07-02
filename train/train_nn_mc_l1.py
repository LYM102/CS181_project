#!/usr/bin/env python3
"""Train L1 gated-state SARSA Q-table."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.engine import GameEngine
from agents.random_agent import RandomAgent
from agents.aggressive_agent import AggressiveAgent
from agents.l1_agent import L1Agent

PROJECT = Path(__file__).parent.parent
BELIEF_PATH = PROJECT / "train/results/policy/belief_net_v4.pt"
SAVE_PATH = PROJECT / "train/results/policy/nn_mc_l1.pt"
SARSA_QTABLE = PROJECT / "sarsa_qtable.pkl"
CFR_POLICY = PROJECT / "cfr_policy.pkl"
COVERAGE_LOG = PROJECT / "logs/l1_coverage.json"


def _train_hands(agent, engine, num_hands, start_idx=0):
    for hand_idx in range(num_hands):
        obs = engine.reset_hand()
        agent.reset()
        state = action = None
        done = False

        while not done:
            cp = engine.current_player
            if cp == agent.player_id:
                state = agent._encode_state(obs)
                action = agent.act(obs)
                agent.record_action(cp, action, obs.current_round)
            else:
                action = engine.agents[cp].act(obs)
                agent.record_action(cp, action, obs.current_round)

            next_obs, reward, done, info = engine.step(action)

            if cp == agent.player_id:
                if done:
                    agent.learn_sarsa(state, action, reward, None, None, done=True)
                elif engine.current_player == agent.player_id:
                    next_state = agent._encode_state(next_obs)
                    next_action = agent.act(next_obs)
                    agent.record_action(agent.player_id, next_action, next_obs.current_round)
                    agent.learn_sarsa(state, action, reward, next_state, next_action, done=False)
                    state, action = next_state, next_action
            obs = next_obs

        agent.decay_epsilon()
        total = start_idx + hand_idx + 1
        if total % 2000 == 0:
            stats = agent.q_table_coverage_stats()
            print(f"  Hand {total} | eps={agent.epsilon:.3f} | Q={stats['total']} "
                  f"(L0={stats['l0_keys']}, belief={stats['belief_keys']}, "
                  f"L0 cov={stats['l0_coverage_pct']:.1f}%)")


def train_l1(num_hands: int = 20000, mix_ratio: float = 0.3,
             belief_path: str = None, save_path: str = None,
             warm_start: bool = True, sarsa_path: str = None,
             confidence: float = 0.65, opp_coarse: str = 'binary',
             l0_backup: float = 0.5):
    belief_path = belief_path or str(BELIEF_PATH)
    save_path = save_path or str(SAVE_PATH)

    agent = L1Agent(
        name="L1_train",
        player_id=0,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9998,
        epsilon_min=0.05,
        state_mode="gated",
        bnn_hidden_dims=(256, 128, 64),
        mc_samples=5,
        bnn_confidence_threshold=confidence,
        gated_opp_coarse=opp_coarse,
        l0_backup_alpha=l0_backup,
    )
    agent._auto_record_self = False
    agent.load_model(belief_path)

    sarsa_path = sarsa_path or str(SARSA_QTABLE)
    if warm_start and Path(sarsa_path).exists():
        agent.warm_start_from_l0(sarsa_path, replicate_belief=True)
    elif warm_start:
        print(f"  Warning: L0 Q-table not found at {sarsa_path}, skipping warm-start")

    phase1_hands = int(num_hands * mix_ratio)
    phase2_hands = num_hands - phase1_hands
    t0 = time.time()

    print(f"L1 config: hands={num_hands}, mix={mix_ratio}, τ={confidence}, "
          f"opp_coarse={opp_coarse}, l0_backup={l0_backup}")

    print(f"Phase 1: {phase1_hands} hands vs Random")
    engine = GameEngine(agent, RandomAgent(name="random_opp"))
    _train_hands(agent, engine, phase1_hands, start_idx=0)
    print(f"  Phase 1 done | {agent.q_table_coverage_stats()}")

    print(f"Phase 2: {phase2_hands} hands vs Aggressive")
    agg_opp = AggressiveAgent(policy_path=str(CFR_POLICY))
    engine = GameEngine(agent, agg_opp)
    _train_hands(agent, engine, phase2_hands, start_idx=phase1_hands)
    stats = agent.q_table_coverage_stats()
    print(f"  Phase 2 done | {stats}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save_model(save_path)

    COVERAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COVERAGE_LOG, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"L1 saved to {save_path} ({time.time() - t0:.1f}s, Q={stats['total']})")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=20000)
    p.add_argument("--mix", type=float, default=0.3,
                   help="Fraction vs Random before Aggressive (default 30/70)")
    p.add_argument("--confidence", type=float, default=0.65,
                   help="BNN confidence threshold for belief state split")
    p.add_argument("--opp-coarse", choices=("3class", "binary", "exploit"),
                   default="binary")
    p.add_argument("--l0-backup", type=float, default=0.5,
                   help="TD error fraction also applied to L0 base key")
    p.add_argument("--belief", type=str, default=str(BELIEF_PATH))
    p.add_argument("--save", type=str, default=str(SAVE_PATH))
    p.add_argument("--warm-start", action="store_true", default=True)
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    p.add_argument("--sarsa", type=str, default=str(SARSA_QTABLE))
    args = p.parse_args()
    train_l1(args.hands, args.mix, args.belief, args.save, args.warm_start,
             args.sarsa, args.confidence, args.opp_coarse, args.l0_backup)


if __name__ == "__main__":
    main()
