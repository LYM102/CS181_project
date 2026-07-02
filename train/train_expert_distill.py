# train/train_expert_distill.py
"""CFR Expert → L3 policy (KL distillation + optional DAgger)."""
from __future__ import annotations

import sys, os, time, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from game.engine import GameEngine
from agents.l3_agent import (
    BNN_PolicyNet, L3Agent,
    collect_expert_policy_data, train_bnn_policy_kl,
)
from agents.expert_agent import ExpertAgent
from agents.random_agent import RandomAgent


def evaluate_vs(agent, opponent_cls, opp_name, num_hands=2000, agent_id=0):
    from game.match_eval import run_match
    opp_factory = lambda: opponent_cls(name=opp_name)
    if agent_id == 0:
        stats = run_match(agent, opp_factory, num_hands=num_hands, report_agent_id=0)
    else:
        stats = run_match(opp_factory, agent, num_hands=num_hands, report_agent_id=1)
    wr = stats.win_rate / 100.0
    wins = stats.wins.get(agent_id, 0)
    losses = stats.num_hands - wins - stats.ties
    print(f"  vs {opp_name:>8s}: WR={wr:.1%} ({wins}W/{losses}L/{stats.ties}T) "
          f"AvgR={stats.avg_reward:+.1f}")
    return wr, stats.avg_reward


def phase1_expert_distill(distill_hands=30000, pretrain_epochs=150,
                          batch_size=64, lr=5e-4,
                          model_save_path="train/results/policy/expert_distill_pretrained.pt",
                          mask_prob=0.5, arch="v2"):
    use_residual = (arch == "v2")
    use_layernorm = (arch == "v2")
    hidden_dims = (256, 128, 64) if arch == "v2" else (128, 64, 32)
    dropout_rate = 0.2 if arch == "v2" else 0.15
    input_dim = 53

    print("=" * 60)
    print("  Phase 1: CFR Expert KL Knowledge Distillation BC")
    print(f"  Architecture: {arch} (input_dim={input_dim}, hidden={hidden_dims})")
    print(f"  Distill hands: {distill_hands}")
    print(f"  Pretrain epochs: {pretrain_epochs}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("\n  Loading Expert with CFR policy...")
    expert = ExpertAgent(name="Expert_Teacher")
    print(f"  CFR info sets: {len(expert.solver.strategy_table)}")

    print(f"\n  Collecting Expert distillation data ({distill_hands} hands vs Random)...")
    t0 = time.time()
    X, y_probs, mask_flags = collect_expert_policy_data(
        expert, num_hands=distill_hands, mask_prob=mask_prob, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  Collected {len(X)} samples in {elapsed:.1f}s ({len(X)/distill_hands:.1f} samples/hand)")

    for i, name in enumerate(["FOLD", "CALL", "RAISE"]):
        avg_p = y_probs[:, i].mean()
        hard_cnt = (y_probs.argmax(axis=1) == i).sum()
        print(f"    {name}: avg_prob={avg_p:.3f}  hard_count={hard_cnt} ({hard_cnt/len(y_probs)*100:.1f}%)")

    print(f"\n  Training with KL Divergence Distillation ({pretrain_epochs} epochs)...")
    model = BNN_PolicyNet(
        input_dim=input_dim, hidden_dims=hidden_dims,
        dropout_rate=dropout_rate, use_residual=use_residual, use_layernorm=use_layernorm,
    ).to(device)

    model = train_bnn_policy_kl(
        model, X, y_probs, mask_flags=mask_flags,
        epochs=pretrain_epochs, batch_size=batch_size, lr=lr,
        alpha=0.3, temperature=3.0, val_split=0.15, device=device, verbose=True)

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    y_hard = y_probs.argmax(axis=1).astype(np.int64)
    torch.save({
        "policy_net_state_dict": model.state_dict(),
        "arch_config": model.get_arch_config(),
        "distill_samples": len(X), "oracle": "expert_cfr",
    }, model_save_path)
    print(f"\n  Pretrained model saved to {model_save_path}")

    bc_data_path = model_save_path.replace(".pt", "_bc_data.npz")
    np.savez_compressed(bc_data_path, X=X, y=y_hard, mask_flags=mask_flags)
    print(f"  BC reference data saved to {bc_data_path}")
    return model, device, X, y_hard, y_probs


def train_one_hand_expert_dagger(env, agent, expert_agent, agent_id=0, reward_filter=True):
    obs = env.reset_hand()
    agent.reset()
    done = False
    step_count = 0
    agent_reward = 0.0
    hand_samples = []
    while not done:
        step_count += 1
        if step_count > 50: break
        cp = env.current_player
        if cp == agent_id:
            features = agent.encode_policy_features(obs)
            action = agent.act(obs)
            expert_probs = expert_agent.get_action_probs(obs)
            hand_samples.append((features, np.array(expert_probs, dtype=np.float32)))
            round_before = obs.current_round
            obs, reward, done, info = env.step(action)
            agent.record_action(cp, action, round_before)
            if done: agent_reward = info.get("result").rewards[agent_id]
        else:
            round_before = obs.current_round
            opp_action = env.agents[cp].act(obs)
            obs, reward, done, info = env.step(opp_action)
            agent.record_action(cp, opp_action, round_before)
            if done: agent_reward = info.get("result").rewards[agent_id]
    if not reward_filter or agent_reward > 0:
        for feat, probs in hand_samples:
            agent.add_dagger_sample(feat, np.argmax(probs))
    agent.decay_epsilon()
    return agent_reward


def phase2_expert_dagger(pretrained_model, device, num_hands=50000,
                         output_path="train/results/policy/expert_distill.pt",
                         dagger_lr=1e-4, dagger_interval=2000, dagger_epochs=10,
                         dagger_capacity=50000, arch="v2"):
    use_residual = (arch == "v2")
    use_layernorm = (arch == "v2")
    hidden_dims = (256, 128, 64) if arch == "v2" else (128, 64, 32)
    dropout_rate = 0.2 if arch == "v2" else 0.15

    print("\n" + "=" * 60)
    print("  Phase 2: DAgger with CFR Expert Oracle")
    print("=" * 60)
    print(f"  Hands: {num_hands} | epsilon: 1.0 -> 0.05")
    print(f"  DAgger LR: {dagger_lr} | interval: {dagger_interval} | epochs: {dagger_epochs}")

    expert = ExpertAgent(name="Expert_Oracle")
    print(f"  CFR info sets: {len(expert.solver.strategy_table)}")

    agent = L3Agent(
        name="BNN_Policy_ExpertDistill", epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05,
        mc_samples=20, device=device,
        hidden_dims=hidden_dims, dropout_rate=dropout_rate,
        use_residual=use_residual, use_layernorm=use_layernorm,
    )
    agent.policy_net.load_state_dict(pretrained_model.state_dict())
    belief_path = "train/results/policy/belief_net.pt"
    if os.path.exists(belief_path):
        agent.load_belief_model(belief_path)
    agent.init_dagger(lr=dagger_lr, capacity=dagger_capacity)

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000
    total_updates = 0

    for hand in range(1, num_hands + 1):
        r = train_one_hand_expert_dagger(env, agent, expert, agent_id=0)
        chips_window += r
        if r > 0: wins_window += 1
        if hand % 1000 == 0:
            print(f"{hand//1000}k ", end="", flush=True)
        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_chips = chips_window / window_size
            wr = wins_window / window_size
            buf_size = len(agent.dagger_buffer)
            print(f"| Hand {hand:>7} | eps={agent.epsilon:.4f} | AvgChips={avg_chips:+.1f} | WR={wr:.1%} | dagger={buf_size} | time={elapsed:.1f}s")
            chips_window = 0.0
            wins_window = 0
        if hand % dagger_interval == 0 and len(agent.dagger_buffer) >= 500:
            d_loss, d_acc = agent.train_dagger(epochs=dagger_epochs, batch_size=128)
            total_updates += 1
            print(f"  [DAgger @ {hand}] buffer={len(agent.dagger_buffer)} loss={d_loss:.4f} acc={d_acc:.3f}")

    total_time = time.time() - start
    print(f"\n  DAgger completed in {total_time:.1f}s ({total_time/60:.1f}min), {total_updates} updates.")
    agent.save_model(output_path)
    return agent


def parse_args():
    raw = sys.argv[1:]
    positional = []
    phase1_only = False
    pretrained_path = None
    arch = "v2"
    i = 0
    while i < len(raw):
        arg = raw[i]
        if arg == "--phase1_only": phase1_only = True
        elif arg == "--pretrained":
            if i + 1 < len(raw):
                pretrained_path = raw[i + 1]; i += 1
        elif arg == "--arch":
            if i + 1 < len(raw):
                arch = raw[i + 1]; i += 1
        elif not arg.startswith("--"): positional.append(arg)
        i += 1
    num_hands = int(positional[0]) if len(positional) > 0 else 50000
    output_path = positional[1] if len(positional) > 1 else "train/results/policy/expert_distill.pt"
    distill_hands = int(positional[2]) if len(positional) > 2 else 30000
    return {"num_hands": num_hands, "output_path": output_path, "distill_hands": distill_hands,
            "phase1_only": phase1_only, "pretrained_path": pretrained_path, "arch": arch}


def main():
    cfg = parse_args()
    print("=" * 60)
    print("  CFR Expert Knowledge Distillation Pipeline")
    print("=" * 60)
    print(f"  Config: {cfg}")
    print("=" * 60)

    os.makedirs(os.path.dirname(cfg["output_path"]) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg["phase1_only"]:
        model, device, X, y, y_probs = phase1_expert_distill(
            distill_hands=cfg["distill_hands"],
            model_save_path=cfg["output_path"], arch=cfg["arch"])
        print("\n" + "=" * 60)
        print("  PHASE 1 EVALUATION")
        print("=" * 60)
        agent = L3Agent(
            name="BNN_Policy_Eval", epsilon=0.0, mc_samples=20, device=device,
            hidden_dims=(256,128,64), dropout_rate=0.2, use_residual=True, use_layernorm=True)
        agent.policy_net.load_state_dict(model.state_dict())
        evaluate_vs(agent, ExpertAgent, "Expert", num_hands=2000, agent_id=0)
        evaluate_vs(agent, RandomAgent, "Random", num_hands=1000, agent_id=0)
        return

    if cfg["pretrained_path"] and os.path.exists(cfg["pretrained_path"]):
        print(f"\n  RESUME from {cfg['pretrained_path']}")
        checkpoint = torch.load(cfg["pretrained_path"], map_location=device)
        model = BNN_PolicyNet(
            input_dim=53, hidden_dims=(256,128,64), dropout_rate=0.2,
            use_residual=True, use_layernorm=True).to(device)
        model.load_state_dict(checkpoint["policy_net_state_dict"])
    else:
        pretrained_path = cfg["output_path"].replace(".pt", "_pretrained.pt")
        model, device, X, y, y_probs = phase1_expert_distill(
            distill_hands=cfg["distill_hands"],
            model_save_path=pretrained_path, arch=cfg["arch"])

    agent = phase2_expert_dagger(
        model, device, num_hands=cfg["num_hands"],
        output_path=cfg["output_path"], arch=cfg["arch"])

    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60)
    agent.epsilon = 0.0
    evaluate_vs(agent, ExpertAgent, "Expert", num_hands=3000, agent_id=0)
    evaluate_vs(agent, RandomAgent, "Random", num_hands=1000, agent_id=0)
    print("\n===== Pipeline Complete =====")


if __name__ == "__main__":
    main()
