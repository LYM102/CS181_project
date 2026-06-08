# train/train_bnn_policy.py — BNN Policy: BC → DAgger → Online RL → Self-Play
"""
Multi-phase BNN Policy training pipeline to surpass SARSA (target: 57%+ WR).

Pipeline:
  Phase 1  - SARSA Behavioral Cloning: (feature, SARSA_action) -> CrossEntropy
  Phase 2  - Multi-round DAgger: play vs Expert, SARSA oracle labels policy states,
             reward-filter (winning hands), periodic fine-tuning. Supports N rounds.
  Phase 3  - Online RL: REINFORCE + EWC anti-forgetting on BC reference
  Phase 4  - Self-Play: train against frozen historical policy snapshots

Usage:
    # Full pipeline
    python -u train/train_bnn_policy.py 100000 output.pt 30000 train/sarsa_final.pkl

    # Skip BC, run DAgger from pretrained
    python -u train/train_bnn_policy.py 100000 output.pt \\
        --skip_phase1 train/results/policy/bnn_policy_bc_pretrained.pt

    # Multi-round DAgger (3 rounds)
    python -u train/train_bnn_policy.py 100000 output.pt \\
        --skip_phase1 pretrained.pt --dagger_rounds 3

    # With Phase 3 (Online RL)
    python -u train/train_bnn_policy.py 50000 output.pt \\
        --skip_phase1 pretrained.pt --phase3

    # V2 architecture (residual + LayerNorm)
    python -u train/train_bnn_policy.py 50000 output.pt \\
        --skip_phase1 pretrained.pt --arch v2

    # Full pipeline: BC + DAgger + RL + SelfPlay
    python -u train/train_bnn_policy.py 150000 output.pt 30000 \
        --dagger_rounds 2 --phase3 --phase4

    # V1→V2 knowledge distillation transfer (accumulative improvement)
    python -u train/train_bnn_policy.py 100000 output.pt \
        --skip_phase1 pretrained_bc.pt \
        --v2_transfer train/results/policy/bnn_policy_dagger.pt \
        --dagger_rounds 2 --phase3
"""
from __future__ import annotations

import sys
import os
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from game.engine import GameEngine
from agents.nn_mc_agent import (
    BNN_PolicyNet, BNN_PolicyAgent,
    collect_policy_distill_data, train_bnn_policy_distill,
    collect_expert_policy_data, train_bnn_policy_kl,
    distill_teacher_to_student,
)
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent


# =========================================================================
#  Evaluation helper
# =========================================================================

def evaluate_vs(agent, opponent_cls, opp_name: str,
                num_hands: int = 2000, agent_id: int = 0):
    """Quick evaluation of agent vs a given opponent type."""
    opp = opponent_cls(name=opp_name)
    env = GameEngine(agent, opp) if agent_id == 0 else GameEngine(opp, agent)
    wins = 0
    losses = 0
    ties = 0
    total_reward = 0.0
    opp_total_reward = 0.0

    for _ in range(num_hands):
        obs = env.reset_hand()
        agent.reset()
        done = False
        step = 0
        while not done:
            step += 1
            if step > 50:
                break
            cp = env.current_player
            if cp == 0:
                action = env.agents[0].act(obs)
            else:
                action = env.agents[1].act(obs)
            round_before = obs.current_round
            obs, reward, done, info = env.step(action)
        if info and "result" in info:
            r0 = info["result"].rewards[0]
            r1 = info["result"].rewards[1]
            total_reward += r0 if agent_id == 0 else r1
            opp_total_reward += r1 if agent_id == 0 else r0
            w = info["result"].winner
            if w == agent_id:
                wins += 1
            elif w == (1 - agent_id):
                losses += 1
            else:
                ties += 1

    wr = wins / num_hands if num_hands > 0 else 0
    avg_r = total_reward / num_hands if num_hands > 0 else 0
    print(f"  vs {opp_name:>8s}: WR={wr:.1%} ({wins}W/{losses}L/{ties}T) "
          f"AvgR={avg_r:+.1f}")
    return wr, avg_r


# =========================================================================
#  Phase 1: SARSA Behavioral Cloning
# =========================================================================

def phase1_distill_and_pretrain(
    distill_hands=30000,
    pretrain_epochs=150,
    batch_size=64,
    lr=5e-4,
    sarsa_model_path="train/sarsa_final.pkl",
    model_save_path="train/results/policy/bnn_policy_pretrained.pt",
    mask_prob=0.5,
    arch="v2",
    oracle="sarsa",
):
    use_residual = (arch == "v2")
    use_layernorm = (arch == "v2")
    hidden_dims = (256, 128, 64) if arch == "v2" else (128, 64, 32)
    dropout_rate = 0.2 if arch == "v2" else 0.15
    input_dim = 53  # upgraded feature dimension

    print("=" * 60)
    print(f"  Phase 1: {'CFR Expert KL Distillation' if oracle == 'expert' else 'SARSA Behavioral Cloning'}")
    print(f"  Oracle:            {oracle}")
    print(f"  Architecture:      {arch} (hidden={hidden_dims}, "
          f"input_dim={input_dim}, residual={use_residual}, layernorm={use_layernorm})")
    print(f"  Distill hands:     {distill_hands}")
    print(f"  Feature masking:   {mask_prob:.0%}")
    print(f"  Pretrain epochs:   {pretrain_epochs}")
    print(f"  Learning rate:     {lr}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    if oracle == "expert":
        # === CFR Expert Knowledge Distillation ===
        print(f"\n  Loading Expert with CFR policy...")
        expert = ExpertAgent(name="Expert_Teacher")
        print(f"  CFR info sets: {len(expert.solver.strategy_table)}")

        print(f"\n  Collecting Expert distillation data ({distill_hands} hands vs Random)...")
        t0 = time.time()
        X, y_probs, mask_flags = collect_expert_policy_data(
            expert, num_hands=distill_hands,
            mask_prob=mask_prob, verbose=True)
        elapsed = time.time() - t0
        print(f"\n  Collected {len(X)} samples in {elapsed:.1f}s")

        # Show Expert's action distribution
        for i, name in enumerate(["FOLD", "CALL", "RAISE"]):
            avg_p = y_probs[:, i].mean()
            hard_cnt = (y_probs.argmax(axis=1) == i).sum()
            print(f"    {name}: avg_prob={avg_p:.3f}  hard_count={hard_cnt}")

        print(f"\n  Training with KL Divergence Distillation ({pretrain_epochs} epochs)...")
        model = BNN_PolicyNet(
            input_dim=input_dim, hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual, use_layernorm=use_layernorm,
        ).to(device)

        model = train_bnn_policy_kl(
            model, X, y_probs, mask_flags=mask_flags,
            epochs=pretrain_epochs, batch_size=batch_size, lr=lr,
            alpha=0.3, temperature=3.0,
            val_split=0.15, device=device, verbose=True)

        # For BC data (EWC), use hard argmax labels
        y = y_probs.argmax(axis=1).astype(np.int64)
    else:
        # === Original SARSA Behavioral Cloning ===
        print(f"\n  Loading SARSA teacher from {sarsa_model_path}...")
        sarsa_agent = SarsaAgent(name="SARSA_teacher", epsilon=0.0,
                                 load_q_table_path=sarsa_model_path)
        print(f"  SARSA Q-table size: {len(sarsa_agent.q_table)} states")

        print(f"\n  Collecting behavioral cloning data ({distill_hands} hands)...")
        t0 = time.time()
        expert_opp = ExpertAgent(name="Expert")
        env = GameEngine(sarsa_agent, expert_opp)

        X, y, mask_flags = collect_policy_distill_data(
            env, sarsa_agent, num_hands=distill_hands,
            mask_prob=mask_prob, verbose=True, observer_player=0)

        elapsed = time.time() - t0
        print(f"\n  Collected {len(X)} samples in {elapsed:.1f}s "
              f"({len(X) / distill_hands:.1f} samples/hand)")

        for i, name in enumerate(["FOLD", "CALL", "RAISE"]):
            cnt = np.sum(y == i)
            print(f"    {name}: {cnt:>6} ({cnt / len(y) * 100:5.1f}%)")
        print(f"  Masked samples: {np.sum(mask_flags)}/{len(X)} = {np.mean(mask_flags):.1%}")

        print(f"\n  Pretraining BNN Policy Network ({pretrain_epochs} epochs)...")
        model = BNN_PolicyNet(
            input_dim=input_dim, hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual, use_layernorm=use_layernorm,
        ).to(device)

        model = train_bnn_policy_distill(
            model, X, y, mask_flags=mask_flags,
            epochs=pretrain_epochs, batch_size=batch_size, lr=lr,
            val_split=0.15, device=device, verbose=True)

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save({
        "policy_net_state_dict": model.state_dict(),
        "arch_config": model.get_arch_config(),
        "distill_samples": len(X),
        "oracle": oracle,
        "sarsa_model": sarsa_model_path if oracle == "sarsa" else None,
    }, model_save_path)
    print(f"\n  Pretrained policy network saved to {model_save_path}")

    # Also save BC reference data for EWC
    bc_data_path = model_save_path.replace(".pt", "_bc_data.npz")
    if oracle == "expert":
        y = y_probs.argmax(axis=1).astype(np.int64)
    np.savez_compressed(bc_data_path, X=X, y=y, mask_flags=mask_flags)
    print(f"  BC reference data saved to {bc_data_path}")

    return model, device, X, y


# =========================================================================
#  Phase 2: Multi-round DAgger
# =========================================================================

def train_one_hand_dagger(env, agent, sarsa_agent, agent_id=0,
                           reward_filter=True):
    """Play one hand with DAgger data collection."""
    obs = env.reset_hand()
    agent.reset()
    done = False
    step_count = 0
    agent_reward = 0.0
    hand_samples = []

    while not done:
        step_count += 1
        if step_count > 50:
            break

        cp = env.current_player

        if cp == agent_id:
            features = agent._feat_builder._encode_bnn_features(obs)
            action = agent.act(obs)

            # SARSA oracle
            sarsa_state = sarsa_agent._encode_state(obs)
            q_vals = sarsa_agent.q_table[sarsa_state]
            legal = obs.legal_actions
            best_val = max(q_vals[a] for a in legal)
            best_actions = [a for a in legal if q_vals[a] == best_val]
            oracle_action = (random.choice(best_actions)
                             if len(best_actions) > 1 else best_actions[0])

            hand_samples.append((features, oracle_action))

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

    # Reward filter: only winning hands
    if not reward_filter or agent_reward > 0:
        for feat, oracle_act in hand_samples:
            agent.add_dagger_sample(feat, oracle_act)

    agent.decay_epsilon()
    return agent_reward


def run_one_dagger_round(agent, sarsa_agent, device, num_hands, round_num,
                          dagger_interval, dagger_epochs, ewc_lambda=0.0):
    """Run one round of DAgger training."""
    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000
    total_dagger_updates = 0

    for hand in range(1, num_hands + 1):
        r = train_one_hand_dagger(env, agent, sarsa_agent, agent_id=0)
        chips_window += r
        if r > 0:
            wins_window += 1

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_chips = chips_window / window_size
            wr = wins_window / window_size
            buf_size = len(agent.dagger_buffer)
            print(
                f"| R{round_num} Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                f"AvgChips={avg_chips:+.1f} | WR={wr:.1%} | "
                f"dagger={buf_size} | time={elapsed:.1f}s"
            )
            chips_window = 0.0
            wins_window = 0

        # Periodic DAgger fine-tuning
        if hand % dagger_interval == 0 and len(agent.dagger_buffer) >= 500:
            d_loss, d_acc = agent.train_dagger(
                epochs=dagger_epochs, batch_size=128, ewc_lambda=ewc_lambda)
            total_dagger_updates += 1
            print(f"  [DAgger R{round_num} @ {hand}] "
                  f"buffer={len(agent.dagger_buffer)} "
                  f"loss={d_loss:.4f} acc={d_acc:.3f}")

    total_time = time.time() - start
    print(f"\n  DAgger Round {round_num} completed in {total_time:.1f}s "
          f"({total_time / 60:.1f}min), {total_dagger_updates} updates.")
    return agent


def phase2_multi_dagger(
    pretrained_model, device, num_hands=50000,
    sarsa_model_path="train/sarsa_final.pkl",
    output_path="train/results/policy/bnn_policy.pt",
    dagger_rounds=1,
    dagger_lr=1e-4,
    dagger_interval=2000,
    dagger_epochs=10,
    dagger_capacity=50000,
    bc_data=None,
    ewc_lambda=0.0,
    arch="v2",
):
    use_residual = (arch == "v2")
    use_layernorm = (arch == "v2")
    hidden_dims = (256, 128, 64) if arch == "v2" else (128, 64, 32)
    dropout_rate = 0.2 if arch == "v2" else 0.15

    print("\n" + "=" * 60)
    print(f"  Phase 2: Multi-Round DAgger ({dagger_rounds} rounds)")
    print("=" * 60)
    print(f"  Architecture:       {arch}")
    print(f"  Hands per round:    {num_hands}")
    print(f"  epsilon schedule:   1.0 -> 0.05 (decay=0.9995)")
    print(f"  DAgger LR:          {dagger_lr}")
    print(f"  Fine-tune interval: every {dagger_interval}")
    print(f"  Fine-tune epochs:   {dagger_epochs}")
    print(f"  Buffer capacity:    {dagger_capacity}")
    print(f"  EWC lambda:         {ewc_lambda}")
    print(f"  Oracle:             SARSA (epsilon=0)")
    print(f"  Output:             {output_path}")
    print("=" * 60)

    print(f"\n  Loading SARSA oracle from {sarsa_model_path}...")
    sarsa_agent = SarsaAgent(name="SARSA_oracle", epsilon=0.0,
                             load_q_table_path=sarsa_model_path)

    # Init policy agent
    agent = BNN_PolicyAgent(
        name="BNN_Policy_DAgger",
        epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05,
        mc_samples=20, device=device,
        hidden_dims=hidden_dims, dropout_rate=dropout_rate,
        use_residual=use_residual, use_layernorm=use_layernorm,
    )
    agent.policy_net.load_state_dict(pretrained_model.state_dict())

    # Setup EWC if using it
    if ewc_lambda > 0 and bc_data is not None:
        X_bc, y_bc = bc_data
        bc_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_bc, dtype=torch.float32),
            torch.tensor(y_bc, dtype=torch.long))
        bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=128, shuffle=True)
        agent.init_ewc(bc_loader, ewc_lambda=ewc_lambda)

    for rnd in range(1, dagger_rounds + 1):
        if rnd > 1:
            # Reset epsilon for new round (with warmer start)
            agent.epsilon = 0.3
            agent.clear_dagger_buffer()

        agent.init_dagger(lr=dagger_lr, capacity=dagger_capacity)
        round_hands = num_hands // dagger_rounds if dagger_rounds > 1 else num_hands

        agent = run_one_dagger_round(
            agent, sarsa_agent, device, round_hands, rnd,
            dagger_interval, dagger_epochs, ewc_lambda=ewc_lambda)

        # Save checkpoint after each round
        round_path = output_path.replace(".pt", f"_r{rnd}.pt")
        agent.save_model(round_path)

        # Quick eval after each round
        print(f"\n  --- Eval after DAgger Round {rnd} ---")
        evaluate_vs(agent, ExpertAgent, "Expert", num_hands=1000, agent_id=0)
        evaluate_vs(agent, RandomAgent, "Random", num_hands=500, agent_id=0)

    agent.save_model(output_path)
    return agent


# =========================================================================
#  Phase 3: Online RL Fine-tuning (REINFORCE + EWC)
# =========================================================================

def train_one_hand_rl(env, agent, agent_id=0):
    """Play one hand collecting (features, action, log_prob, reward)."""
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

    # Assign hand reward to all steps
    for s in agent.rl_trajectory:
        s["reward"] = agent_reward

    agent.decay_epsilon()
    return agent_reward


def phase3_online_rl(
    agent, device, num_hands=50000,
    rl_lr=3e-5, ewc_lambda=100.0, update_freq=10,
    bc_data=None, output_path="train/results/policy/bnn_policy_rl.pt",
):
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

    # Setup EWC on BC reference data
    if ewc_lambda > 0 and bc_data is not None:
        X_bc, y_bc = bc_data
        bc_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_bc, dtype=torch.float32),
            torch.tensor(y_bc, dtype=torch.long))
        bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=128, shuffle=True)
        agent.init_ewc(bc_loader, ewc_lambda=ewc_lambda)

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000
    total_rl_loss = 0.0
    total_rl_steps = 0

    # Use low epsilon for RL (mostly on-policy)
    agent.epsilon = 0.05

    for hand in range(1, num_hands + 1):
        r = train_one_hand_rl(env, agent, agent_id=0)
        chips_window += r
        if r > 0:
            wins_window += 1

        # REINFORCE update
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
            print(
                f"| RL Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                f"AvgChips={avg_chips:+.1f} | WR={wr:.1%} | "
                f"RL_loss={avg_loss:.4f} | time={elapsed:.1f}s"
            )
            chips_window = 0.0
            wins_window = 0

    total_time = time.time() - start
    print(f"\nPhase 3 completed in {total_time:.1f}s ({total_time / 60:.1f}min).")
    agent.save_model(output_path)
    return agent


# =========================================================================
#  Phase 4: Self-Play
# =========================================================================

def phase4_selfplay(
    agent, device, num_hands=50000,
    rl_lr=1e-5, ewc_lambda=50.0, update_freq=20,
    bc_data=None, output_path="train/results/policy/bnn_policy_sp.pt",
):
    print("\n" + "=" * 60)
    print("  Phase 4: Self-Play Training")
    print("=" * 60)
    print(f"  Total hands:        {num_hands}")
    print(f"  RL learning rate:   {rl_lr}")
    print(f"  EWC lambda:         {ewc_lambda}")
    print(f"  Update frequency:   every {update_freq} hands")
    print("=" * 60)

    agent.init_rl_optimizer(lr=rl_lr)

    # EWC on BC reference to prevent forgetting
    if ewc_lambda > 0 and bc_data is not None:
        X_bc, y_bc = bc_data
        bc_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_bc, dtype=torch.float32),
            torch.tensor(y_bc, dtype=torch.long))
        bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=128, shuffle=True)
        agent.init_ewc(bc_loader, ewc_lambda=ewc_lambda)

    # Create frozen opponent from current policy snapshot
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

    # Alternate who is agent[0] (learner) to collect diverse experiences
    start = time.time()
    chips_window = 0.0
    wins_window = 0
    window_size = 1000
    total_rl_loss = 0.0
    total_rl_steps = 0

    agent.epsilon = 0.02  # very low exploration in self-play

    for hand in range(1, num_hands + 1):
        # Alternate: half the time learner is P0, half P1
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

        # Assign reward
        for s in agent.rl_trajectory:
            s["reward"] = learner_reward

        chips_window += learner_reward
        if learner_reward > 0:
            wins_window += 1

        # REINFORCE update
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
            print(
                f"| SP Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                f"AvgChips={avg_chips:+.1f} | WR={wr:.1%} | "
                f"RL_loss={avg_loss:.4f} | time={elapsed:.1f}s"
            )
            chips_window = 0.0
            wins_window = 0

            # Update opponent snapshot periodically
            if hand % (window_size * 5) == 0:
                opp_agent.policy_net.load_state_dict(
                    {k: v.clone() for k, v in agent.policy_net.state_dict().items()})
                print(f"  [SelfPlay @ {hand}] Opponent snapshot updated")

    total_time = time.time() - start
    print(f"\nPhase 4 completed in {total_time:.1f}s ({total_time / 60:.1f}min).")
    agent.save_model(output_path)
    return agent


# =========================================================================
#  Main
# =========================================================================

def parse_args():
    """Robust argument parsing with flags and positional args."""
    raw = sys.argv[1:]

    # Flags
    skip_phase1 = False
    pretrain_path_override = None
    dagger_rounds = 1
    run_phase3 = False
    run_phase4 = False
    arch = "v2"
    sarsa_model_override = None
    v2_transfer = None  # path to V1 teacher for distillation transfer

    # Positional args
    positional = []

    i = 0
    while i < len(raw):
        arg = raw[i]
        if arg == "--skip_phase1":
            skip_phase1 = True
            if i + 1 < len(raw) and not raw[i + 1].startswith("--"):
                pretrain_path_override = raw[i + 1]
                i += 1
        elif arg == "--dagger_rounds":
            if i + 1 < len(raw):
                dagger_rounds = int(raw[i + 1])
                i += 1
        elif arg == "--phase3":
            run_phase3 = True
        elif arg == "--phase4":
            run_phase4 = True
        elif arg == "--arch":
            if i + 1 < len(raw):
                arch = raw[i + 1]
                i += 1
        elif arg == "--sarsa":
            if i + 1 < len(raw):
                sarsa_model_override = raw[i + 1]
                i += 1
        elif arg == "--v2_transfer":
            if i + 1 < len(raw):
                v2_transfer = raw[i + 1]
                i += 1
        elif not arg.startswith("--"):
            positional.append(arg)
        i += 1

    num_hands = int(positional[0]) if len(positional) > 0 else 50000
    output_path = positional[1] if len(positional) > 1 else "train/results/policy/bnn_policy.pt"
    distill_hands = int(positional[2]) if len(positional) > 2 else 30000
    sarsa_model = sarsa_model_override or (positional[3] if len(positional) > 3 else "train/sarsa_final.pkl")
    pretrain_path = (pretrain_path_override or
                     (positional[4] if len(positional) > 4
                      else "train/results/policy/bnn_policy_pretrained.pt"))

    return {
        "num_hands": num_hands,
        "output_path": output_path,
        "distill_hands": distill_hands,
        "sarsa_model": sarsa_model,
        "pretrain_path": pretrain_path,
        "skip_phase1": skip_phase1,
        "dagger_rounds": dagger_rounds,
        "run_phase3": run_phase3,
        "run_phase4": run_phase4,
        "arch": arch,
        "v2_transfer": v2_transfer,
    }


def main():
    cfg = parse_args()

    print("=" * 60)
    print("  BNN Policy Training Pipeline")
    print("=" * 60)
    print(f"  Config: {cfg}")
    print("=" * 60)

    os.makedirs(os.path.dirname(cfg["output_path"]) or ".", exist_ok=True)

    bc_data = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dims = (256, 128, 64) if cfg["arch"] == "v2" else (128, 64, 32)
    dropout_rate = 0.2 if cfg["arch"] == "v2" else 0.15
    use_residual = (cfg["arch"] == "v2")
    use_layernorm = (cfg["arch"] == "v2")

    # ---- Phase 1: BC Pretrain ----
    if cfg["skip_phase1"] and os.path.exists(cfg["pretrain_path"]):
        print("\n" + "=" * 60)
        print("  RESUME MODE: Skipping Phase 1")
        print(f"  Loading: {cfg['pretrain_path']}")
        print("=" * 60)

        checkpoint = torch.load(cfg["pretrain_path"], map_location=device)
        # Detect arch from checkpoint; fallback to V1 for legacy checkpoints
        if "arch_config" in checkpoint:
            arch_cfg = checkpoint["arch_config"]
            hidden_dims = arch_cfg.get("hidden_dims", hidden_dims)
            dropout_rate = arch_cfg.get("dropout_rate", dropout_rate)
            use_residual = arch_cfg.get("use_residual", False)
            use_layernorm = arch_cfg.get("use_layernorm", False)
        else:
            # Legacy V1 checkpoint: (128,64,32), no residual/LayerNorm
            print("  [Legacy checkpoint detected, using V1 architecture]")
            hidden_dims = (128, 64, 32)
            dropout_rate = 0.15
            use_residual = False
            use_layernorm = False
            cfg["arch"] = "v1"

        model = BNN_PolicyNet(
            input_dim=53, hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual, use_layernorm=use_layernorm,
        ).to(device)
        model.load_state_dict(checkpoint["policy_net_state_dict"])
        print(f"  Loaded (distill_samples={checkpoint.get('distill_samples', '?')})")

        # Try to load BC reference data for EWC
        bc_data_path = cfg["pretrain_path"].replace(".pt", "_bc_data.npz")
        if os.path.exists(bc_data_path):
            bc_npz = np.load(bc_data_path)
            bc_data = (bc_npz["X"], bc_npz["y"])
            print(f"  BC reference data: {len(bc_data[0])} samples")
    else:
        model, device, X, y = phase1_distill_and_pretrain(
            distill_hands=cfg["distill_hands"],
            sarsa_model_path=cfg["sarsa_model"],
            model_save_path=cfg["pretrain_path"],
            arch=cfg["arch"],
        )
        bc_data = (X, y)

    # ---- Phase 1.5: V1→V2 Knowledge Distillation Transfer ----
    if cfg["v2_transfer"] and os.path.exists(cfg["v2_transfer"]):
        print("\n" + "=" * 60)
        print("  Phase 1.5: V1→V2 Knowledge Distillation Transfer")
        print(f"  Teacher: {cfg['v2_transfer']}")
        print(f"  Target arch: V2 (256,128,64) residual+layernorm")
        print("=" * 60)

        # Load V1 teacher
        teacher_ckpt = torch.load(cfg["v2_transfer"], map_location=device)
        teacher_arch = teacher_ckpt.get("arch_config", {})
        teacher_hidden = teacher_arch.get("hidden_dims", (128, 64, 32))
        teacher_dropout = teacher_arch.get("dropout_rate", 0.15)

        teacher = BNN_PolicyNet(
            input_dim=53, hidden_dims=teacher_hidden,
            dropout_rate=teacher_dropout,
            use_residual=teacher_arch.get("use_residual", False),
            use_layernorm=teacher_arch.get("use_layernorm", False),
        ).to(device)
        teacher.load_state_dict(teacher_ckpt["policy_net_state_dict"])
        print(f"  Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")

        # Need BC data for distillation
        if bc_data is None:
            bc_data_path = cfg["pretrain_path"].replace(".pt", "_bc_data.npz")
            if os.path.exists(bc_data_path):
                bc_npz = np.load(bc_data_path)
                bc_data = (bc_npz["X"], bc_npz["y"])
                print(f"  Loaded BC data: {len(bc_data[0])} samples")
            else:
                print("  [WARN] No BC data for distillation, collecting...")
                sarsa_agent = SarsaAgent(name="SARSA_collect", epsilon=0.0,
                                         load_q_table_path=cfg["sarsa_model"])
                expert = ExpertAgent()
                env = GameEngine(sarsa_agent, expert)
                X, y, _ = collect_policy_distill_data(
                    env, sarsa_agent, num_hands=cfg["distill_hands"],
                    mask_prob=0.5, verbose=True, observer_player=0)
                bc_data = (X, y)
                np.savez_compressed(bc_data_path, X=X, y=y)

        # Create V2 student
        student = BNN_PolicyNet(
            input_dim=53, hidden_dims=(256, 128, 64),
            dropout_rate=0.2, use_residual=True, use_layernorm=True,
        ).to(device)
        print(f"  Student created: {sum(p.numel() for p in student.parameters()):,} params")

        # Distill
        student = distill_teacher_to_student(
            teacher, student,
            bc_data[0], bc_data[1],
            epochs=100, alpha=0.5, temperature=3.0,
            device=device, verbose=True,
        )

        # Save distilled V2 model
        distill_path = cfg["output_path"].replace(".pt", "_distill.pt")
        torch.save({
            "policy_net_state_dict": student.state_dict(),
            "arch_config": student.get_arch_config(),
            "teacher_model": cfg["v2_transfer"],
        }, distill_path)
        print(f"  Distilled V2 model saved to {distill_path}")

        # Replace model with V2 student for subsequent phases
        model = student
        cfg["arch"] = "v2"

    # ---- Phase 2: Multi-round DAgger ----
    agent = phase2_multi_dagger(
        model, device,
        num_hands=cfg["num_hands"],
        sarsa_model_path=cfg["sarsa_model"],
        output_path=cfg["output_path"],
        dagger_rounds=cfg["dagger_rounds"],
        bc_data=bc_data,
        arch=cfg["arch"],
    )

    # ---- Phase 3: Online RL + EWC ----
    if cfg["run_phase3"]:
        rl_output = cfg["output_path"].replace(".pt", "_rl.pt")
        agent = phase3_online_rl(
            agent, device,
            num_hands=cfg["num_hands"] // 2,
            bc_data=bc_data,
            output_path=rl_output,
        )

    # ---- Phase 4: Self-Play ----
    if cfg["run_phase4"]:
        sp_output = cfg["output_path"].replace(".pt", "_sp.pt")
        agent = phase4_selfplay(
            agent, device,
            num_hands=cfg["num_hands"] // 2,
            bc_data=bc_data,
            output_path=sp_output,
        )

    # ---- Final Evaluation ----
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60)
    agent.epsilon = 0.0  # no exploration for eval
    evaluate_vs(agent, ExpertAgent, "Expert", num_hands=3000, agent_id=0)
    evaluate_vs(agent, RandomAgent, "Random", num_hands=1000, agent_id=0)

    print("\n===== Pipeline Complete =====")


if __name__ == "__main__":
    main()
