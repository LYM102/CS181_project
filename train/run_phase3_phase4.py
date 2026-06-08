# train/run_phase3_phase4.py — Run Phase 3 (Online RL) + Phase 4 (Self-Play)
# on top of Expert Distill checkpoint.
"""
Usage:
    python -u train/run_phase3_phase4.py \
        train/results/policy/expert_distill.pt \
        --phase3 --phase4 \
        --bc_data train/results/policy/expert_distill_pretrained_bc_data.npz
"""
from __future__ import annotations

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from game.engine import GameEngine
from agents.nn_mc_agent import BNN_PolicyNet, BNN_PolicyAgent
from agents.expert_agent import ExpertAgent
from agents.random_agent import RandomAgent


def evaluate_vs(agent, opponent_cls, opp_name, num_hands=2000, agent_id=0):
    opp = opponent_cls(name=opp_name)
    env = GameEngine(agent, opp) if agent_id == 0 else GameEngine(opp, agent)
    wins = losses = ties = 0
    total_reward = 0.0
    for _ in range(num_hands):
        obs = env.reset_hand()
        chips_before = env.players[agent_id].chips
        agent.reset()
        done = False
        step = 0
        while not done:
            step += 1
            if step > 50:
                break
            cp = env.current_player
            action = env.agents[cp].act(obs)
            obs, reward, done, info = env.step(action)
        if info and "result" in info:
            # True zero-sum: net chip change instead of engine rewards
            total_reward += env.players[agent_id].chips - chips_before
            w = info["result"].winner
            if w == agent_id: wins += 1
            elif w == (1 - agent_id): losses += 1
            else: ties += 1
    wr = wins / num_hands if num_hands > 0 else 0
    avg_r = total_reward / num_hands if num_hands > 0 else 0
    print(f"  vs {opp_name:>8s}: WR={wr:.1%} ({wins}W/{losses}L/{ties}T) AvgR={avg_r:+.1f}")
    return wr, avg_r


# =========================================================================
#  Phase 3: Online RL Fine-tuning (REINFORCE + EWC)
# =========================================================================

def train_one_hand_rl(env, agent, agent_id=0):
    obs = env.reset_hand()
    agent.reset()
    done = False
    step_count = 0
    agent_reward = 0.0

    while not done:
        step_count += 1
        if step_count > 50:
            break
        cp = env.current_player
        if cp == agent_id:
            features = agent._feat_builder._encode_bnn_features(obs)
            action, log_prob = agent.act_with_logprob(obs)
            agent.record_rl_step(features, action, log_prob, 0.0)
            round_before = obs.current_round
            obs, reward, done, info = env.step(action)
            agent.record_action(cp, action, round_before)
            if done:
                agent_reward = info.get("result").rewards[agent_id]
        else:
            round_before = obs.current_round
            opp_action = env.agents[cp].act(obs)
            obs, reward, done, info = env.step(opp_action)
            agent.record_action(cp, opp_action, round_before)
            if done:
                agent_reward = info.get("result").rewards[agent_id]

    for s in agent.rl_trajectory:
        s["reward"] = agent_reward
    agent.decay_epsilon()
    return agent_reward


def phase3_online_rl(agent, device, num_hands=50000,
                     rl_lr=3e-5, ewc_lambda=100.0, update_freq=10,
                     bc_data=None, output_path="train/results/policy/expert_distill_rl.pt"):
    print("\n" + "=" * 60)
    print("  Phase 3: Online RL Fine-tuning (REINFORCE + EWC)")
    print("=" * 60)
    print(f"  Total hands:        {num_hands}")
    print(f"  RL learning rate:   {rl_lr}")
    print(f"  EWC lambda:         {ewc_lambda}")
    print(f"  Update frequency:   every {update_freq} hands")
    print(f"  epsilon:            {agent.epsilon:.4f} (fixed-low)")
    print("=" * 60)

    agent.init_rl_optimizer(lr=rl_lr)

    if ewc_lambda > 0 and bc_data is not None:
        X_bc, y_bc = bc_data
        bc_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_bc, dtype=torch.float32),
            torch.tensor(y_bc, dtype=torch.long))
        bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=128, shuffle=True)
        agent.init_ewc(bc_loader, ewc_lambda=ewc_lambda)
        print(f"  EWC initialized on {len(X_bc)} BC samples")

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000
    total_rl_loss = 0.0
    total_rl_steps = 0
    agent.epsilon = 0.05

    for hand in range(1, num_hands + 1):
        r = train_one_hand_rl(env, agent, agent_id=0)
        chips_window += r
        if r > 0:
            wins_window += 1

        if hand % update_freq == 0:
            rl_loss, n_steps = agent.train_rl_step(ewc_lambda=ewc_lambda)
            total_rl_loss += rl_loss
            total_rl_steps += n_steps

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_chips = chips_window / window_size
            wr = wins_window / window_size
            avg_loss = total_rl_loss / max(total_rl_steps, 1)
            print(f"| RL Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                  f"AvgChips={avg_chips:+.1f} | WR={wr:.1%} | "
                  f"RL_loss={avg_loss:.4f} | time={elapsed:.1f}s")
            chips_window = 0.0
            wins_window = 0

    total_time = time.time() - start
    print(f"\n  Phase 3 completed in {total_time:.1f}s ({total_time / 60:.1f}min).")
    agent.save_model(output_path)
    return agent


# =========================================================================
#  Phase 4: Self-Play
# =========================================================================

def phase4_selfplay(agent, device, num_hands=50000,
                    rl_lr=1e-5, ewc_lambda=50.0, update_freq=20,
                    bc_data=None, output_path="train/results/policy/expert_distill_sp.pt",
                    epsilon=0.02, opp_update_every=5000):
    print("\n" + "=" * 60)
    print("  Phase 4: Self-Play Training")
    print("=" * 60)
    print(f"  Total hands:        {num_hands}")
    print(f"  RL learning rate:   {rl_lr}")
    print(f"  EWC lambda:         {ewc_lambda}")
    print(f"  Update frequency:   every {update_freq} hands")
    print("=" * 60)

    agent.init_rl_optimizer(lr=rl_lr)

    if ewc_lambda > 0 and bc_data is not None:
        X_bc, y_bc = bc_data
        bc_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_bc, dtype=torch.float32),
            torch.tensor(y_bc, dtype=torch.long))
        bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=128, shuffle=True)
        agent.init_ewc(bc_loader, ewc_lambda=ewc_lambda)

    opp_agent = BNN_PolicyAgent(
        name="SelfPlay_Opponent",
        epsilon=0.0, device=device,
        hidden_dims=agent._hidden_dims,
        dropout_rate=agent._dropout_rate,
        use_residual=agent._use_residual,
        use_layernorm=agent._use_layernorm,
    )
    opp_agent.policy_net.load_state_dict(
        {k: v.clone() for k, v in agent.policy_net.state_dict().items()})

    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000
    total_rl_loss = 0.0
    total_rl_steps = 0
    agent.epsilon = epsilon

    for hand in range(1, num_hands + 1):
        swap = (hand % 2 == 0)
        if swap:
            env = GameEngine(opp_agent, agent)
            learner_id = 1
        else:
            env = GameEngine(agent, opp_agent)
            learner_id = 0

        obs = env.reset_hand()
        agent.reset()
        opp_agent.reset()
        done = False
        step_count = 0
        learner_reward = 0.0

        while not done:
            step_count += 1
            if step_count > 50:
                break
            cp = env.current_player
            if cp == learner_id:
                features = agent._feat_builder._encode_bnn_features(obs)
                action, log_prob = agent.act_with_logprob(obs)
                agent.record_rl_step(features, action, log_prob, 0.0)
                round_before = obs.current_round
                obs, reward, done, info = env.step(action)
                agent.record_action(cp, action, round_before)
                if done:
                    learner_reward = info.get("result").rewards[learner_id]
            else:
                round_before = obs.current_round
                opp_action = opp_agent.act(obs)
                obs, reward, done, info = env.step(opp_action)
                opp_agent.record_action(cp, opp_action, round_before)
                if done:
                    learner_reward = info.get("result").rewards[learner_id]

        for s in agent.rl_trajectory:
            s["reward"] = learner_reward

        chips_window += learner_reward
        if learner_reward > 0:
            wins_window += 1

        if hand % update_freq == 0:
            rl_loss, n_steps = agent.train_rl_step(ewc_lambda=ewc_lambda)
            total_rl_loss += rl_loss
            total_rl_steps += n_steps

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_chips = chips_window / window_size
            wr = wins_window / window_size
            avg_loss = total_rl_loss / max(total_rl_steps, 1)
            print(f"| SP Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                  f"AvgChips={avg_chips:+.1f} | WR={wr:.1%} | "
                  f"RL_loss={avg_loss:.4f} | time={elapsed:.1f}s")
            chips_window = 0.0
            wins_window = 0

            if hand % opp_update_every == 0:
                opp_agent.policy_net.load_state_dict(
                    {k: v.clone() for k, v in agent.policy_net.state_dict().items()})
                print(f"  [SelfPlay @ {hand}] Opponent snapshot updated")

    total_time = time.time() - start
    print(f"\n  Phase 4 completed in {total_time:.1f}s ({total_time / 60:.1f}min).")
    agent.save_model(output_path)
    return agent


# =========================================================================
#  Main
# =========================================================================

def parse_args():
    raw = sys.argv[1:]
    run_phase3 = False
    run_phase4 = False
    bc_data_path = None
    num_hands = 50000
    epsilon = 0.02
    opp_update_every = 5000
    positional = []

    i = 0
    while i < len(raw):
        arg = raw[i]
        if arg == "--phase3":
            run_phase3 = True
        elif arg == "--phase4":
            run_phase4 = True
        elif arg == "--bc_data":
            if i + 1 < len(raw):
                bc_data_path = raw[i + 1]; i += 1
        elif arg == "--hands":
            if i + 1 < len(raw):
                num_hands = int(raw[i + 1]); i += 1
        elif arg == "--eps":
            if i + 1 < len(raw):
                epsilon = float(raw[i + 1]); i += 1
        elif arg == "--opp_update":
            if i + 1 < len(raw):
                opp_update_every = int(raw[i + 1]); i += 1
        elif not arg.startswith("--"):
            positional.append(arg)
        i += 1

    if not run_phase3 and not run_phase4:
        run_phase3 = True
        run_phase4 = True

    model_path = positional[0] if len(positional) > 0 else "train/results/policy/expert_distill.pt"
    return {
        "model_path": model_path,
        "run_phase3": run_phase3,
        "run_phase4": run_phase4,
        "bc_data_path": bc_data_path,
        "num_hands": num_hands,
        "epsilon": epsilon,
        "opp_update_every": opp_update_every,
    }


def main():
    cfg = parse_args()

    print("=" * 60)
    print("  Phase 3+4: Online RL + Self-Play")
    print(f"  Base model: {cfg['model_path']}")
    print(f"  Phase 3: {'YES' if cfg['run_phase3'] else 'NO'}")
    print(f"  Phase 4: {'YES' if cfg['run_phase4'] else 'NO'}")
    print(f"  Hands:    {cfg['num_hands']}")
    print("=" * 60)

    os.makedirs("train/results/policy", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load checkpoint
    print(f"\n  Loading model from {cfg['model_path']}...")
    checkpoint = torch.load(cfg["model_path"], map_location=device)

    if "arch_config" in checkpoint:
        arch_cfg = checkpoint["arch_config"]
        hidden_dims = arch_cfg.get("hidden_dims", (256, 128, 64))
        dropout_rate = arch_cfg.get("dropout_rate", 0.2)
        use_residual = arch_cfg.get("use_residual", True)
        use_layernorm = arch_cfg.get("use_layernorm", True)
    else:
        hidden_dims = (256, 128, 64)
        dropout_rate = 0.2
        use_residual = True
        use_layernorm = True

    print(f"  Arch: hidden={hidden_dims}, dropout={dropout_rate}, "
          f"residual={use_residual}, layernorm={use_layernorm}")
    print(f"  Samples: {checkpoint.get('distill_samples', '?')}, "
          f"oracle: {checkpoint.get('oracle', '?')}")

    # Load BC data for EWC
    bc_data = None
    bc_path = cfg.get("bc_data_path")
    if bc_path and os.path.exists(bc_path):
        bc_npz = np.load(bc_path)
        bc_data = (bc_npz["X"], bc_npz["y"])
        print(f"  BC data loaded: {len(bc_data[0])} samples from {bc_path}")
    else:
        # Try auto-detect
        auto_path = cfg["model_path"].replace(".pt", "_pretrained_bc_data.npz")
        if os.path.exists(auto_path):
            bc_npz = np.load(auto_path)
            bc_data = (bc_npz["X"], bc_npz["y"])
            print(f"  BC data auto-loaded: {len(bc_data[0])} samples")
        else:
            print("  [WARN] No BC data for EWC — running without anti-forgetting")

    # Create agent
    agent = BNN_PolicyAgent(
        name="BNN_Policy_RL",
        epsilon=0.05, epsilon_decay=0.9995, epsilon_min=0.05,
        mc_samples=20, device=device,
        hidden_dims=hidden_dims, dropout_rate=dropout_rate,
        use_residual=use_residual, use_layernorm=use_layernorm,
    )
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])

    # Quick baseline eval
    print("\n  --- Baseline Evaluation ---")
    evaluate_vs(agent, ExpertAgent, "Expert", num_hands=2000, agent_id=0)
    evaluate_vs(agent, RandomAgent, "Random", num_hands=1000, agent_id=0)

    # Phase 3
    if cfg["run_phase3"]:
        rl_output = cfg["model_path"].replace(".pt", "_rl.pt")
        agent = phase3_online_rl(
            agent, device,
            num_hands=cfg["num_hands"],
            bc_data=bc_data,
            output_path=rl_output,
        )
        print("\n  --- After Phase 3 ---")
        evaluate_vs(agent, ExpertAgent, "Expert", num_hands=2000, agent_id=0)
        evaluate_vs(agent, RandomAgent, "Random", num_hands=1000, agent_id=0)

    # Phase 4
    if cfg["run_phase4"]:
        sp_output = (rl_output if cfg["run_phase3"] else cfg["model_path"]).replace(".pt", "_sp.pt")
        agent = phase4_selfplay(
            agent, device,
            num_hands=cfg["num_hands"],
            bc_data=bc_data,
            output_path=sp_output,
            epsilon=cfg["epsilon"],
            opp_update_every=cfg["opp_update_every"],
        )
        print("\n  --- After Phase 4 ---")
        evaluate_vs(agent, ExpertAgent, "Expert", num_hands=2000, agent_id=0)
        evaluate_vs(agent, RandomAgent, "Random", num_hands=1000, agent_id=0)

    # Final Evaluation
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION (3000 hands vs Expert)")
    print("=" * 60)
    agent.epsilon = 0.0
    evaluate_vs(agent, ExpertAgent, "Expert", num_hands=3000, agent_id=0)
    evaluate_vs(agent, RandomAgent, "Random", num_hands=1000, agent_id=0)

    print("\n===== Phase 3+4 Pipeline Complete =====")


if __name__ == "__main__":
    main()
