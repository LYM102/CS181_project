#!/usr/bin/env python3
"""Train residual gate g_θ (L2 or L3 logits)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.l1_agent import L1Agent
from agents.l3_agent import L3Agent
from agents.belief_gating import (
    collect_gating_data,
    collect_gating_data_from_q_agent,
    train_gating_net,
    DEFAULT_GATING_PATH,
)

PROJECT = Path(__file__).parent.parent
BELIEF_PATH = PROJECT / "train/results/policy/belief_net_v4.pt"
POLICY_PATH = PROJECT / "train/results/policy/expert_distill_v2.pt"
L1_PATH = PROJECT / "train/results/policy/nn_mc_l1.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--policy", type=str, default=str(POLICY_PATH))
    p.add_argument("--belief", type=str, default=str(BELIEF_PATH))
    p.add_argument("--save", type=str, default=str(PROJECT / DEFAULT_GATING_PATH))
    p.add_argument("--from-l1", action="store_true",
                   help="Collect oracle data from L1 Q-logits")
    p.add_argument("--l1", type=str, default=str(L1_PATH))
    p.add_argument("--exploit-oversample", type=int, default=8,
                   help="Repeat exploit (oracle≠base) samples during gate training")
    p.add_argument("--init", type=str, default=None,
                   help="Warm-start gate weights from checkpoint (e.g. bak_l2)")
    args = p.parse_args()

    if args.from_l1:
        agent = L1Agent(
            name="GatingTrainL1", player_id=0, epsilon=0.0, state_mode="gated",
            use_action_gating=False, bnn_hidden_dims=(256, 128, 64),
        )
        agent._auto_record_self = False
        agent.load_model(args.l1)
        print(f"Collecting gating data from L1 Q-agent ({args.hands} hands vs Aggressive)...")
        X, y = collect_gating_data_from_q_agent(agent, num_hands=args.hands)
    else:
        agent = L3Agent(
            name="GatingTrain",
            player_id=0,
            epsilon=0.0,
            use_belief=True,
            use_learned_gating=False,
        )
        agent.load_model(args.policy)
        agent.load_belief_model(args.belief)
        print(f"Collecting gating data from L3 ({args.hands} hands vs Aggressive)...")
        X, y = collect_gating_data(agent, num_hands=args.hands)
    print(f"  Samples: {len(X)}")
    for c in range(3):
        print(f"    class {c}: {(y == c).sum()}")

    train_gating_net(X, y, epochs=args.epochs, save_path=args.save,
                     exploit_oversample=args.exploit_oversample,
                     init_path=args.init)


if __name__ == "__main__":
    main()
