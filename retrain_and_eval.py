"""Retrain CFR/SARSA and optional belief/L3/gate/L1 pipeline."""

import argparse
import pickle
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from game.engine import GameEngine
from agents.random_agent import RandomAgent
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent
from game.cfr_solver import CFRSolver, equity_to_bucket
from game.card import build_full_deck
from game.evaluator import equity_to_bin, compute_hand_strength, HAND_STRENGTH_SAMPLES
from game.match_eval import run_match

FULL_DECK = build_full_deck()

PROJECT = Path(__file__).parent
CFR_PATH = PROJECT / "cfr_policy.pkl"
SARSA_PATH = PROJECT / "sarsa_qtable.pkl"
BELIEF_PATH = PROJECT / "train/results/policy/belief_net_v4.pt"
POLICY_PATH = PROJECT / "train/results/policy/expert_distill_v2.pt"
GATING_PATH = PROJECT / "train/results/policy/belief_gating.pt"
L1_PATH = PROJECT / "train/results/policy/nn_mc_l1.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Retrain CFR/SARSA + progressive agents")
    p.add_argument("--clean", action="store_true", help="Remove stale logs/models first")
    p.add_argument("--skip-cfr", action="store_true")
    p.add_argument("--skip-sarsa", action="store_true")
    p.add_argument("--cfr-iters", type=int, default=30000)
    p.add_argument("--sarsa-hands", type=int, default=10000)
    p.add_argument("--train-belief", action="store_true")
    p.add_argument("--belief-hands", type=int, default=5000)
    p.add_argument("--belief-epochs", type=int, default=50)
    p.add_argument("--train-distill", action="store_true")
    p.add_argument("--distill-hands", type=int, default=30000)
    p.add_argument("--train-gating", action="store_true")
    p.add_argument("--gating-hands", type=int, default=8000)
    p.add_argument("--gating-epochs", type=int, default=80)
    p.add_argument("--train-l1", action="store_true")
    p.add_argument("--l1-hands", type=int, default=10000)
    p.add_argument("--eval-seeds", type=int, default=3, help="Seeds for quick eval (time-saving)")
    return p.parse_args()


def run_sarsa_hand(engine, learner, learner_id: int = 0):
    """One SARSA episode: learner only updates on its own decisions."""
    for a in engine.agents:
        if hasattr(a, "reset"):
            a.reset()
    obs = engine.reset_hand()
    state = action = None
    done = False

    while not done:
        cp = engine.current_player
        if cp == learner_id:
            state = learner._encode_state(obs)
            action = learner.act(obs)
        else:
            action = engine.agents[cp].act(obs)

        next_obs, reward, done, info = engine.step(action)

        if cp == learner_id:
            if done:
                learner.learn(state, action, reward, None, None, done=True)
            elif engine.current_player == learner_id:
                next_state = learner._encode_state(next_obs)
                next_action = learner.act(next_obs)
                learner.learn(state, action, reward, next_state, next_action, done=False)
                state, action = next_state, next_action
        obs = next_obs


def main():
    args = parse_args()

    # Line-buffer stdout when piped (tee / log files)
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    print("=" * 70)
    print("  RETRAIN + EVALUATE (treys, learnable gating pipeline)")
    print("=" * 70)

    if args.clean:
        print("\n[0] Clean stale experiment data")
        import shutil
        logs = PROJECT / "logs"
        if logs.is_dir():
            for p in logs.iterdir():
                if p.name in ("RESULTS_FINAL.json", "progressive_ladder.json",
                              "progressive_l3.json", "strict_ablation.json",
                              "viz_data"):
                    continue
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)

    if not args.skip_cfr:
        print(f"\n[1] Retrain CFR ({args.cfr_iters} iterations)")
        t0 = time.time()
        solver = CFRSolver()
        stats = solver.train(iterations=args.cfr_iters, log_interval=5000)
        solver.save_policy(str(CFR_PATH))
        print(f"  CFR: {stats['info_sets']} info sets ({time.time() - t0:.1f}s)")
    else:
        print("\n[1] Skip CFR")

    if not args.skip_sarsa:
        print(f"\n[2] Train SARSA vs Random ({args.sarsa_hands} hands)")
        t0 = time.time()
        sarsa = SarsaAgent(
            name="sarsa_train", alpha=0.1, gamma=0.95,
            epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05)
        engine = GameEngine(sarsa, RandomAgent(name="random_opp"))
        for hand_idx in range(args.sarsa_hands):
            run_sarsa_hand(engine, sarsa, learner_id=0)
            sarsa.decay_epsilon()
            if (hand_idx + 1) % 2000 == 0:
                print(f"  Hand {hand_idx + 1}/{args.sarsa_hands} | "
                      f"eps={sarsa.epsilon:.3f} | Q={sarsa.get_q_table_size()}")
        sarsa.save_q_table(str(SARSA_PATH))
        print(f"  SARSA: Q={sarsa.get_q_table_size()} ({time.time() - t0:.1f}s)")
    else:
        print("\n[2] Skip SARSA")

    print("\n[3] Abstraction diagnostics")
    print("-" * 50)
    strengths = []
    bin_counts = defaultdict(int)
    bucket_counts = defaultdict(int)
    for combo in combinations(FULL_DECK, 2):
        hs = compute_hand_strength(list(combo), [], num_samples=HAND_STRENGTH_SAMPLES)
        strengths.append(hs)
        bin_counts[equity_to_bin(hs, bins=20)] += 1
        bucket_counts[equity_to_bucket(hs, 10)] += 1
    used_bins = sum(1 for b in range(20) if bin_counts.get(b, 0) > 0)
    used_buckets = sum(1 for b in range(10) if bucket_counts.get(b, 0) > 0)
    print(f"  Preflop strength: [{min(strengths):.4f}, {max(strengths):.4f}]")
    print(f"  SARSA bins: {used_bins}/20 | CFR buckets: {used_buckets}/10")
    if CFR_PATH.exists():
        with open(CFR_PATH, "rb") as f:
            print(f"  CFR info sets: {len(pickle.load(f).get('strategy_table', {}))}")
    if SARSA_PATH.exists():
        sq = SarsaAgent(name="e", load_q_table_path=str(SARSA_PATH))
        print(f"  SARSA Q-table: {sq.get_q_table_size()}")

    if SARSA_PATH.exists() and CFR_PATH.exists():
        seeds = list(range(42, 42 + args.eval_seeds))
        print(f"\n[4] Quick eval ({len(seeds)} seeds × 1000 hands)")
        for name, mk0, mk1 in [
            ("SARSA vs CFR",
             lambda: SarsaAgent(name="s", load_q_table_path=str(SARSA_PATH)),
             lambda: ExpertAgent(name="cfr")),
            ("Random vs CFR",
             lambda: RandomAgent(name="r"),
             lambda: ExpertAgent(name="cfr")),
            ("SARSA vs Random",
             lambda: SarsaAgent(name="s", load_q_table_path=str(SARSA_PATH)),
             lambda: RandomAgent(name="r")),
        ]:
            wrs, avgs = [], []
            for seed in seeds:
                a0, a1 = mk0(), mk1()
                if hasattr(a0, "epsilon"):
                    a0.epsilon = 0.0
                stats = run_match(a0, a1, num_hands=1000, seed=seed, report_agent_id=0)
                wrs.append(stats.win_rate)
                avgs.append(stats.avg_reward)
            m, s = sum(wrs) / len(wrs), (sum((x - sum(wrs) / len(wrs)) ** 2 for x in wrs) / len(wrs)) ** 0.5
            print(f"  {name}: WR {m:.1f}% ± {s:.1f} | AvgR {sum(avgs) / len(avgs):+.2f}")

    if args.train_belief:
        if not CFR_PATH.exists():
            raise SystemExit("Need cfr_policy.pkl for BNN training")
        print("\n[5] Train BNN belief model")
        from train.train_belief_net import train_belief_net
        t0 = time.time()
        train_belief_net(
            num_hands=args.belief_hands, epochs=args.belief_epochs,
            save_path=str(BELIEF_PATH))
        print(f"  BNN → {BELIEF_PATH} ({time.time() - t0:.1f}s)")

    if args.train_distill:
        if not CFR_PATH.exists():
            raise SystemExit("Need cfr_policy.pkl for distillation")
        print(f"\n[6] Distill L3 policy ({args.distill_hands} hands)")
        from train.train_expert_distill import phase1_expert_distill
        t0 = time.time()
        phase1_expert_distill(
            distill_hands=args.distill_hands,
            model_save_path=str(POLICY_PATH),
            arch="v2",
        )
        print(f"  Policy → {POLICY_PATH} ({time.time() - t0:.1f}s)")

    if args.train_gating:
        if not POLICY_PATH.exists() or not BELIEF_PATH.exists():
            raise SystemExit("Need expert_distill_v2.pt and belief_net_v4.pt for gating")
        print(f"\n[7] Train learnable gating ({args.gating_hands} hands)")
        from agents.l3_agent import L3Agent
        from agents.belief_gating import collect_gating_data, train_gating_net
        agent = L3Agent(name="gt", player_id=0, epsilon=0.0,
                                use_belief=True, use_learned_gating=False)
        agent.load_model(str(POLICY_PATH))
        agent.load_belief_model(str(BELIEF_PATH))
        X, y = collect_gating_data(agent, num_hands=args.gating_hands)
        train_gating_net(X, y, epochs=args.gating_epochs, save_path=str(GATING_PATH))

    if args.train_l1:
        if not BELIEF_PATH.exists():
            raise SystemExit("Need belief_net_v4.pt for L1")
        print(f"\n[8] Train L1 Belief-Augmented SARSA ({args.l1_hands} hands)")
        from train.train_nn_mc_l1 import train_l1
        train_l1(args.l1_hands, belief_path=str(BELIEF_PATH), save_path=str(L1_PATH))

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
