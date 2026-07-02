"""Ladder eval: L0–L3 vs Random / Aggressive / CFR."""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

from game.match_eval import run_match

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from game.engine import GameEngine
from game.constants import FOLD, CALL, RAISE
from game.match_eval import reset_eval_stacks
from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent
from agents.expert_agent import ExpertAgent
from agents.aggressive_agent import AggressiveAgent
from agents.l1_agent import L1Agent
from agents.l2_agent import L2Agent
from agents.l3_agent import L3Agent

BELIEF = PROJECT / "train/results/policy/belief_net_v4.pt"
GATING = PROJECT / "train/results/policy/belief_gating.pt"
GATING_L2 = PROJECT / "train/results/policy/belief_gating_l2.pt"
GATING_L3 = PROJECT / "train/results/policy/belief_gating_l3.pt"
POLICY = PROJECT / "train/results/policy/expert_distill_v2.pt"
L1 = PROJECT / "train/results/policy/nn_mc_l1.pt"
SARSA_QTABLE = PROJECT / "sarsa_qtable.pkl"
CFR_POLICY = PROJECT / "cfr_policy.pkl"


def load_l0():
    a = SarsaAgent(name="L0", load_q_table_path=str(SARSA_QTABLE))
    a.epsilon = 0.0
    return a


def _warmup_qtable(agent, num_hands=2000, seed=12345):
    """Warm up Q-table by running SARSA training vs Random so L1/L2 have reasonable policies."""
    from agents.random_agent import RandomAgent
    old_eps = agent.epsilon
    agent.epsilon = 1.0
    agent.epsilon_decay = 0.999
    agent.epsilon_min = 0.05
    random.seed(seed)
    np.random.seed(seed)
    opp = RandomAgent(name="WarmupOpp")
    engine = GameEngine(agent, opp)
    for hand_idx in range(num_hands):
        obs = engine.reset_hand()
        agent.reset()
        if hasattr(opp, 'reset'):
            opp.reset()
        state = action = None
        done = False
        while not done:
            cp = engine.current_player
            if cp == agent.player_id:
                state = agent._encode_state(obs)
                action = agent.act(obs)
            else:
                action = engine.agents[cp].act(obs)
            next_obs, reward, done, info = engine.step(action)
            if cp == agent.player_id:
                if done:
                    agent.learn_sarsa(state, action, reward, None, None, done=True)
                elif engine.current_player == agent.player_id:
                    next_state = agent._encode_state(next_obs)
                    next_action = agent.act(next_obs)
                    agent.learn_sarsa(state, action, reward, next_state, next_action, done=False)
                    state, action = next_state, next_action
            obs = next_obs
        agent.decay_epsilon()
    agent.epsilon = 0.0  # greedy for evaluation
    print(f"  Q-table warmup: {agent.get_q_table_size()} states after {num_hands} hands")


def load_l1():
    a = L1Agent(name="L1", player_id=0, epsilon=1.0, state_mode="gated",
                bnn_hidden_dims=(256, 128, 64))
    a._auto_record_self = False
    a.load_model(str(L1 if L1.exists() else BELIEF))
    if a.get_q_table_size() < 500:
        _warmup_qtable(a, num_hands=500)
    else:
        print(f"  L1: using trained Q-table ({a.get_q_table_size()} states), skip warmup")
    a.epsilon = 0.0  # greedy for evaluation
    return a


def _resolve_l2_gate() -> str:
    for p in (GATING_L2,
              PROJECT / "train/results/policy/belief_gating.pt.bak_l2"):
        if p.exists():
            return str(p)
    return str(GATING)


def _resolve_l3_gate() -> str:
    for p in (GATING_L3,
              PROJECT / "train/results/policy/belief_gating_l2.pt",
              PROJECT / "train/results/policy/belief_gating.pt.bak_l2"):
        if p.exists():
            return str(p)
    return str(GATING)


def load_l2():
    a = L2Agent(name="L2", player_id=0, epsilon=1.0, state_mode="gated",
                gating_model_path=_resolve_l2_gate(),
                bnn_hidden_dims=(256, 128, 64))
    a._auto_record_self = False
    a.load_model(str(L1 if L1.exists() else BELIEF))
    if a.get_q_table_size() < 500:
        _warmup_qtable(a, num_hands=500)
    else:
        print(f"  L2: using trained Q-table ({a.get_q_table_size()} states), skip warmup")
    a.epsilon = 0.0  # greedy for evaluation
    a.bluff_log = []
    return a


def load_l3(belief_gate: bool = True, gate_selective: bool = True,
            gate_scale: float = 0.5, deterministic_belief: bool = False):
    a = L3Agent(
        name="L3", player_id=0, epsilon=0.0,
        use_belief=belief_gate,
        use_learned_gating=belief_gate,
        gating_model_path=_resolve_l3_gate() if belief_gate else None,
        gate_selective=gate_selective,
        gate_scale=gate_scale,
        deterministic_belief=deterministic_belief,
    )
    a.load_model(str(POLICY))
    if belief_gate:
        a.load_belief_model(str(BELIEF))
    a.bluff_log = []
    return a


# Bluff/trap ground-truth thresholds (Aggressive opponent labeling)
BLUFF_EQUITY_THRESHOLD = 0.3
TRAP_EQUITY_THRESHOLD = 0.6


class TrackedAggressive(AggressiveAgent):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._trace = []

    def reset(self):
        self._trace = []

    def act(self, obs):
        s = obs.equity
        act = super().act(obs)
        self._trace.append({
            "is_bluff": s < BLUFF_EQUITY_THRESHOLD and act == RAISE,
            "is_trap": s > TRAP_EQUITY_THRESHOLD and act == CALL,
        })
        return act

    def pop_gt(self):
        t, self._trace = self._trace, []
        return t


def run_hands(agent, opponent_factory, num_hands: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    agent0 = agent() if callable(agent) else agent
    opp = opponent_factory()
    if hasattr(agent0, "reset"):
        pass
    engine = GameEngine(agent0, opp)
    wins = defaultdict(int)
    total_r = defaultdict(float)
    bluff_stats = {"tp": 0, "fp": 0, "fn": 0, "trap_tp": 0, "trap_fp": 0, "trap_fn": 0}

    tracked = isinstance(opp, TrackedAggressive)
    for _ in range(num_hands):
        reset_eval_stacks(engine)
        if hasattr(agent0, "bluff_log"):
            log_start = len(agent0.bluff_log)

        result = engine.run_hand()
        if result.winner is not None:
            wins[result.winner] += 1
        for pid in range(2):
            total_r[pid] += result.rewards[pid]

        if tracked and hasattr(agent0, "bluff_log"):
            entries = agent0.bluff_log[log_start:]
            gt = opp.pop_gt()
            det_b = any(e.get("bluff_raise") for e in entries)
            det_t = any(e.get("slow_play_trap") for e in entries)
            if not det_t:
                for e in entries:
                    belief_strong = e.get("belief_strong", 0.0)
                    equity = e.get("equity", 0.5)
                    if belief_strong > 0.2 and equity > 0.35:
                        det_t = True
                        break
            act_b = any(g["is_bluff"] for g in gt)
            act_t = any(g["is_trap"] for g in gt)
            bluff_stats["tp"] += int(det_b and act_b)
            bluff_stats["fp"] += int(det_b and not act_b)
            bluff_stats["fn"] += int(not det_b and act_b)
            bluff_stats["trap_tp"] += int(det_t and act_t)
            bluff_stats["trap_fp"] += int(det_t and not act_t)
            bluff_stats["trap_fn"] += int(not det_t and act_t)

    wr = wins[0] / num_hands * 100
    avgr = total_r[0] / num_hands
    out = {"win_rate": wr, "avg_reward": avgr}
    if tracked:
        def pr(tp, fp):
            return tp / (tp + fp) * 100 if tp + fp > 0 else 0.0
        def re(tp, fn):
            return tp / (tp + fn) * 100 if tp + fn > 0 else 0.0
        out["bluff_precision"] = pr(bluff_stats["tp"], bluff_stats["fp"])
        out["bluff_recall"] = re(bluff_stats["tp"], bluff_stats["fn"])
        out["trap_precision"] = pr(bluff_stats["trap_tp"], bluff_stats["trap_fp"])
        out["trap_recall"] = re(bluff_stats["trap_tp"], bluff_stats["trap_fn"])
    return out


def eval_agent(agent_loader, opponents, seeds, hands):
    results = {}
    for opp_name, opp_fn in opponents.items():
        wrs, avgs = [], []
        extra = defaultdict(list)
        for seed in seeds:
            r = run_hands(agent_loader, opp_fn, hands, seed)
            wrs.append(r["win_rate"])
            avgs.append(r["avg_reward"])
            for k in ("bluff_precision", "bluff_recall", "trap_precision", "trap_recall"):
                if k in r:
                    extra[k].append(r[k])
        results[opp_name] = {
            "win_rate_mean": float(np.mean(wrs)),
            "win_rate_std": float(np.std(wrs)),
            "avg_reward_mean": float(np.mean(avgs)),
            "avg_reward_std": float(np.std(avgs)),
        }
        for k, vals in extra.items():
            results[opp_name][f"{k}_mean"] = float(np.mean(vals))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    seeds = list(range(42, 42 + args.seeds))

    opponents = {
        "Random": lambda: RandomAgent(name="Random"),
        "Aggressive": lambda: TrackedAggressive(policy_path=str(CFR_POLICY)),
        "CFR": lambda: ExpertAgent(name="CFR"),
    }

    agents = {
        "L0_SARSA": load_l0,
        "L1_BeliefSARSA": load_l1,
        "L2_BeliefSARSA_Gate": load_l2,
        "L3_BeliefGate": lambda: load_l3(True),
        "L3_NoGate": lambda: load_l3(False),
    }

    print("=" * 70)
    print(f"Progressive eval: {args.seeds} seeds × {args.hands} hands")
    print("=" * 70)

    all_results = {}
    for name, loader in agents.items():
        print(f"\n--- {name} ---")
        try:
            res = eval_agent(loader, opponents, seeds, args.hands)
            all_results[name] = res
            for opp, m in res.items():
                print(f"  vs {opp:12s}: WR {m['win_rate_mean']:.1f}±{m['win_rate_std']:.1f}%  "
                      f"AvgR {m['avg_reward_mean']:+.2f}")
                if "bluff_recall_mean" in m:
                    print(f"               bluff P/R {m.get('bluff_precision_mean', 0):.0f}/"
                          f"{m['bluff_recall_mean']:.0f}%")
        except Exception as e:
            print(f"  SKIP: {e}")
            all_results[name] = {"error": str(e)}

    out = PROJECT / "logs/progressive_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"seeds": seeds, "hands": args.hands, "results": all_results}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
