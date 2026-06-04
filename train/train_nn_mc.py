# train_nn_mc.py — Train NN_MCAgent (BNN + SARSA Q-table) for 52-card Texas Hold'em

"""
Training pipeline for NN_MCAgent:

  Phase 1 – BNN Training (high-quality data from Expert vs trained SARSA):
    1. Use ExpertAgent vs trained SarsaAgent to generate meaningful gameplay.
    2. Record BOTH self and opponent actions for rich BNN features.
    3. Collect 20k+ hands → ~50k+ samples with validation split.

  Phase 2 – SARSA Q-table Training (TD learning):
    1. Use the trained BNN to predict opponent strength at each decision.
    2. Build Q-table state = (H_code, B_level, Pot_bin, O_NN).
    3. SARSA (TD) updates: every agent step bootstraps from Q(s',a').
       This replaces the high-variance MC updates used previously.
    4. Record BOTH self and opponent actions (consistent with BNN training).

Usage:
    conda activate fmd
    python train/train_nn_mc.py [num_hands] [output_path] [bnn_data_hands] [sarsa_model] [mask_prob]

Examples:
    python train/train_nn_mc.py 50000 train/nn_mc_model.pt 20000 train/sarsa_final.pkl 0.5
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from game.engine import GameEngine
from agents.nn_mc_agent import (
    NN_MCAgent, BNNWithMCDropout,
    collect_bnn_training_data, train_bnn,
)
from agents.random_agent import RandomAgent
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent


# =========================================================================
#  Phase 1: Collect BNN training data + Train BNN
# =========================================================================

def phase1_collect_and_train_bnn(
    data_hands=20000,
    bnn_epochs=200,
    bnn_batch_size=64,
    bnn_lr=1e-3,
    sarsa_model_path="train/sarsa_final.pkl",
    bnn_save_path="train/bnn_pretrained.pt",
    mask_prob=0.5,  # 50% mask: forces BNN to learn from public features
):
    print("=" * 60)
    print("  Phase 1: BNN Pre-training (Expert vs Trained SARSA)")
    print(f"  Feature masking: {mask_prob:.0%} of opponent features masked")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Collecting {data_hands} hands from Expert vs SARSA...")

    t0 = time.time()
    sarsa_agent = SarsaAgent(name="SARSA_p0", epsilon=0.0,
                             load_q_table_path=sarsa_model_path)
    expert_agent = ExpertAgent(name="Expert_p1")
    env = GameEngine(sarsa_agent, expert_agent)

    X, y, mask_flags = collect_bnn_training_data(
        env, num_hands=data_hands, mask_prob=mask_prob, verbose=True)
    elapsed = time.time() - t0
    print(f"  Collected {len(X)} labeled samples in {elapsed:.1f}s "
          f"(masked={np.sum(mask_flags)}/{len(X)}={np.mean(mask_flags):.1%})")
    print(f"  Label distribution: weak={np.sum(y == 0)}, "
          f"mid={np.sum(y == 1)}, strong={np.sum(y == 2)}")

    print(f"\n  Training BNN ({bnn_epochs} epochs, val_split=0.2)...")
    model = BNNWithMCDropout(input_dim=47).to(device)
    model = train_bnn(model, X, y, mask_flags=mask_flags, epochs=bnn_epochs,
                      batch_size=bnn_batch_size, lr=bnn_lr,
                      val_split=0.2, device=device, verbose=True)

    torch.save({
        "bnn_state_dict": model.state_dict(),
        "bnn_trained": True,
    }, bnn_save_path)
    print(f"  BNN saved to {bnn_save_path}")
    return model, device


# =========================================================================
#  Phase 2: SARSA Q-table training (TD learning instead of MC)
# =========================================================================

def train_one_hand_sarsa_bnn(env, agent, agent_id=0):
    """
    Train one hand using SARSA (TD) updates with BNN-augmented state.

    Follows the same pattern as train_sarsa.py's train_one_hand:
      - Agent's turn: act + SARSA update on previous (s,a) pair
      - Opponent's turn: let opponent decide, no agent learning
      - Terminal reward from info["result"].rewards[agent_id]

    Key difference from MC: updates happen every agent step (TD bootstrapping)
    instead of only at episode end, reducing variance significantly.
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
                agent.learn_sarsa(prev_state, prev_action, 0.0,
                                  state, action, done=False)

            round_before = obs.current_round
            obs, reward, done, info = env.step(action)
            agent.record_action(cp, action, round_before)

            if done:
                agent_reward = info.get("result").rewards[agent_id]
                agent.learn_sarsa(state, action, agent_reward,
                                  None, None, done=True)
            else:
                prev_state, prev_action = state, action
        else:
            # ===== Opponent's turn =====
            round_before = obs.current_round
            opp_action = env.agents[cp].act(obs)
            obs, reward, done, info = env.step(opp_action)
            agent.record_action(cp, opp_action, round_before)

            if done and prev_state is not None:
                # Opponent ended the hand — terminal update for agent's last (s,a)
                agent_reward = info.get("result").rewards[agent_id]
                agent.learn_sarsa(prev_state, prev_action, agent_reward,
                                  None, None, done=True)

    agent.decay_epsilon()
    return agent_reward


def _compute_gated_stats(agent: NN_MCAgent) -> float:
    """Return fraction of Q-table states that are BNN-augmented (non-sentinel)."""
    if len(agent.q_table) == 0:
        return 0.0
    gated_count = sum(1 for state in agent.q_table if len(state) >= 4 and state[3] != -1)
    return gated_count / len(agent.q_table)


def _finetune_bnn_online(agent: NN_MCAgent, X: np.ndarray, y: np.ndarray,
                          device: str, epochs: int = 5, lr: float = 5e-5):
    """
    Lightweight online fine-tuning of the BNN on recent data collected
    during Phase 2 play. Uses a very low LR to avoid catastrophic forgetting.
    """
    model = agent.bnn_model
    model.train()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # Class weights based on recent data distribution
    num_classes = model.num_classes
    class_counts = np.bincount(y, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def phase2_train_sarsa_qtable(
    bnn_model, device, num_hands=50000,
    bnn_model_path="train/bnn_pretrained.pt",
    output_path="train/nn_mc_model.pt",
    state_mode='gated',
    bnn_finetune_interval=5000,
):
    print("\n" + "=" * 60)
    print("  Phase 2: SARSA Q-table Training (gated BNN state + TD learning)")
    print("=" * 60)
    print(f"  Total hands:  {num_hands}")
    print(f"  State mode:   {state_mode}")
    print(f"  ε_min:        0.05 (reduced for less noise)")
    print(f"  BNN finetune: every {bnn_finetune_interval} hands")
    print(f"  Output:       {output_path}")

    agent = NN_MCAgent(
        name="NN_MC",
        epsilon=1.0, epsilon_decay=0.9998, epsilon_min=0.05,
        alpha=0.1, gamma=0.95, mc_samples=20, device=device,
        state_mode=state_mode,
    )
    agent.bnn_model = bnn_model
    agent.bnn_trained = True
    agent._auto_record_self = False

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    # --- Accumulators for BNN online fine-tuning ---
    ft_X, ft_y = [], []

    start = time.time()
    total_reward_window = 0.0
    wins_window = 0
    chips_window = 0.0
    window_size = 1000

    for hand in range(1, num_hands + 1):
        r = train_one_hand_sarsa_bnn(env, agent, agent_id=0)
        total_reward_window += r
        chips_window += r
        if r > 0:
            wins_window += 1

        # --- Collect BNN fine-tuning data from recent hands ---
        # Only use showdown hands where opponent's true label is known
        if agent.bnn_trained and hasattr(env, 'players'):
            try:
                opp = env.players[1]
                if not opp.folded and len(env.community_cards) >= 3:
                    obs_for_ft = env._get_observation(0)
                    opp_hole = opp.hole_cards
                    if len(opp_hole) == 2:
                        from treys import Card
                        opp_ranks = [Card.get_rank_int(c) for c in opp_hole]
                        opp_rank_avg = sum(opp_ranks) / (len(opp_ranks) * 12.0)
                        opp_suited = 1.0 if Card.get_suit_int(opp_hole[0]) == Card.get_suit_int(opp_hole[1]) else 0.0
                        from game.evaluator import compute_equity
                        opp_eq = compute_equity(opp_hole, env.community_cards)
                        from agents.nn_mc_agent import _equity_to_strength_label
                        label = _equity_to_strength_label(opp_eq)
                        feat = agent._encode_bnn_features(obs_for_ft)  # masked (inference-mode)
                        ft_X.append(feat)
                        ft_y.append(label)
            except Exception:
                pass

        if hand % 1000 == 0:
            print(f"{hand // 1000}k ", end="", flush=True)

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_r = total_reward_window / window_size
            wr = wins_window / window_size
            avg_chips = chips_window / window_size
            gated_pct = _compute_gated_stats(agent) if state_mode == 'gated' else 0
            print(
                f"| Hand {hand:>7} | eps={agent.epsilon:.4f} | "
                f"Qsize={agent.get_q_table_size():>5} | "
                f"AvgR={avg_r:+.2f} | AvgChips={avg_chips:+.1f} | "
                f"WR={wr:.1%} | time={elapsed:.1f}s"
                + (f" | gated={gated_pct:.0%}" if state_mode == 'gated' else "")
            )
            total_reward_window = 0.0
            wins_window = 0
            chips_window = 0.0

        # --- Periodic BNN online fine-tuning ---
        if bnn_finetune_interval > 0 and hand % bnn_finetune_interval == 0 and len(ft_X) >= 200:
            n_ft = min(len(ft_X), 3000)
            X_ft = np.array(ft_X[-n_ft:], dtype=np.float32)
            y_ft = np.array(ft_y[-n_ft:], dtype=np.int64)
            _finetune_bnn_online(agent, X_ft, y_ft, device, epochs=5, lr=5e-5)
            # Keep a sliding window of recent data
            ft_X = ft_X[-2000:]
            ft_y = ft_y[-2000:]

    agent.save_model(output_path)
    total_time = time.time() - start
    print(f"\nTraining completed in {total_time:.1f}s ({total_time / 60:.1f}min).")
    print(f"Final Q-table size: {agent.get_q_table_size()}")


# =========================================================================
#  Main
# =========================================================================

def main():
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "train/nn_mc_model.pt"
    bnn_data_hands = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    sarsa_model = sys.argv[4] if len(sys.argv) > 4 else "train/sarsa_final.pkl"
    mask_prob = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5  # 50% mask default
    bnn_path = "train/bnn_pretrained.pt"

    bnn_model, device = phase1_collect_and_train_bnn(
        data_hands=bnn_data_hands,
        bnn_epochs=200,
        sarsa_model_path=sarsa_model,
        bnn_save_path=bnn_path,
        mask_prob=mask_prob,
    )

    phase2_train_sarsa_qtable(
        bnn_model=bnn_model,
        device=device,
        num_hands=num_hands,
        bnn_model_path=bnn_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
