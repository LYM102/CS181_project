# train/experiment_ablation.py — Ablation experiments for NN_MC agent
"""
Run multiple experiments with different configurations to find optimal setup.

Experiments:
  A. Compact state space (argmax + confidence) → 1080 total states
  B. Larger BNN (256,128,64) + dropout=0.2 + CosineAnnealing + label smoothing
  C. Best combined: compact state + larger BNN

Usage:
    conda activate fmd
    python train/experiment_ablation.py [experiment_id]  # A, B, C, or ALL
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from game.engine import GameEngine
from agents.nn_mc_agent import (
    NN_MCAgent, BNNWithMCDropout,
    collect_bnn_training_data, train_bnn,
)
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent


# =========================================================================
#  Enhanced BNN training with label smoothing + CosineAnnealing
# =========================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better generalization."""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # One-hot with smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        if self.weight is not None:
            # Apply class weights
            sample_weights = self.weight[targets]
            loss = -(smooth_targets * log_probs).sum(dim=-1)
            loss = (loss * sample_weights).mean()
        else:
            loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


def train_bnn_enhanced(model, X, y, mask_flags=None,
                       epochs=200, batch_size=64, lr=1e-3,
                       val_split=0.2, device="cpu", verbose=True,
                       use_label_smoothing=True, use_cosine=True):
    """
    Enhanced BNN training with:
    - Label smoothing (0.1)
    - CosineAnnealingWarmRestarts scheduler
    - Gradient clipping
    - AdamW with weight decay
    """
    n = len(X)
    n_val = int(n * val_split)
    indices = np.random.RandomState(42).permutation(n)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_t = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_t = torch.tensor(y[train_idx], dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if n_val > 0:
        X_v = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        y_v = torch.tensor(y[val_idx], dtype=torch.long).to(device)

    # Class weights
    y_train_np = y[train_idx]
    class_counts = np.bincount(y_train_np, minlength=3)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler: CosineAnnealingWarmRestarts or ReduceLROnPlateau
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-6)

    # Loss: label smoothing or standard cross entropy
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights_t)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    if verbose:
        print(f"  Enhanced training: label_smoothing={use_label_smoothing}, "
              f"cosine_schedule={use_cosine}")
        print(f"  Class weights: {class_weights}")

    model.train()
    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        train_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0

        # Scheduler step
        if use_cosine:
            scheduler.step()
        else:
            val_acc = 0.0
            if n_val > 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_v)
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == y_v).float().mean().item()
                model.train()
            scheduler.step(val_acc)

        # Validation
        val_acc = 0.0
        if n_val > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_v)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_v).float().mean().item()
            model.train()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1:>3}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"TrainAcc: {train_acc:.3f} | ValAcc: {val_acc:.3f} | "
                  f"Best: {best_val_acc:.3f} | LR: {current_lr:.2e}")

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if verbose:
        print(f"  Final Best ValAcc: {best_val_acc:.3f} (restored best weights)")

    return model


# =========================================================================
#  Experiment runner
# =========================================================================

EXPERIMENT_CONFIGS = {
    'A': {
        'name': 'Compact State (argmax+conf)',
        'state_mode': 'compact',
        'bnn_hidden_dims': (128, 64, 32),
        'bnn_dropout': 0.15,
        'enhanced_train': False,
        'description': 'Same BNN, compact state space (1080 vs 4860)',
    },
    'B': {
        'name': 'Larger BNN + Enhanced Training',
        'state_mode': 'compact',
        'bnn_hidden_dims': (256, 128, 64),
        'bnn_dropout': 0.2,
        'enhanced_train': True,
        'description': 'Bigger BNN (256-128-64), dropout=0.2, label smoothing, cosine LR',
    },
    'C': {
        'name': 'Argmax State + Larger BNN',
        'state_mode': 'argmax',
        'bnn_hidden_dims': (256, 128, 64),
        'bnn_dropout': 0.2,
        'enhanced_train': True,
        'description': 'Simplest state (540 total), bigger BNN, enhanced training',
    },
}


def run_experiment(exp_id, bnn_data_hands=20000, sarsa_hands=30000,
                   sarsa_model_path="train/sarsa_final.pkl", mask_prob=0.5):
    """Run a single experiment configuration."""
    config = EXPERIMENT_CONFIGS[exp_id]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"train/results/exp_{exp_id}_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  EXPERIMENT {exp_id}: {config['name']}")
    print(f"  {config['description']}")
    print(f"  State mode: {config['state_mode']}, BNN: {config['bnn_hidden_dims']}")
    print(f"  Dropout: {config['bnn_dropout']}, Enhanced: {config['enhanced_train']}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ---- Phase 1: Collect BNN data + train ----
    print(f"\n  [Phase 1] Collecting {bnn_data_hands} hands for BNN training...")
    t0 = time.time()

    sarsa_agent = SarsaAgent(name="SARSA_p0", epsilon=0.0,
                             load_q_table_path=sarsa_model_path)
    expert_agent = ExpertAgent(name="Expert_p1")
    env = GameEngine(sarsa_agent, expert_agent)

    X, y, mask_flags = collect_bnn_training_data(
        env, num_hands=bnn_data_hands, mask_prob=mask_prob, verbose=True)
    elapsed = time.time() - t0
    print(f"  Collected {len(X)} samples in {elapsed:.1f}s "
          f"(masked={np.mean(mask_flags):.1%})")

    # Build BNN model with experiment-specific architecture
    bnn_model = BNNWithMCDropout(
        input_dim=47,
        hidden_dims=config['bnn_hidden_dims'],
        dropout_rate=config['bnn_dropout'],
    ).to(device)

    print(f"\n  [Phase 1] Training BNN...")
    if config['enhanced_train']:
        bnn_model = train_bnn_enhanced(
            bnn_model, X, y, mask_flags=mask_flags,
            epochs=200, batch_size=64, lr=1e-3,
            val_split=0.2, device=device, verbose=True,
            use_label_smoothing=True, use_cosine=True,
        )
    else:
        bnn_model = train_bnn(
            bnn_model, X, y, mask_flags=mask_flags,
            epochs=200, batch_size=64, lr=1e-3,
            val_split=0.2, device=device, verbose=True,
        )

    bnn_path = os.path.join(out_dir, "bnn_pretrained.pt")
    torch.save({"bnn_state_dict": bnn_model.state_dict(), "bnn_trained": True}, bnn_path)

    # ---- Phase 2: SARSA Q-table training ----
    print(f"\n  [Phase 2] Training SARSA Q-table ({sarsa_hands} hands)...")
    from train.train_nn_mc import train_one_hand_sarsa_bnn

    agent = NN_MCAgent(
        name=f"NN_MC_{exp_id}",
        epsilon=1.0, epsilon_decay=0.9998, epsilon_min=0.10,
        alpha=0.1, gamma=0.95, mc_samples=20, device=device,
        state_mode=config['state_mode'],
        bnn_hidden_dims=config['bnn_hidden_dims'],
        bnn_dropout=config['bnn_dropout'],
    )
    agent.bnn_model = bnn_model
    agent.bnn_trained = True
    agent._auto_record_self = False

    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)

    start = time.time()
    total_reward_window = 0.0
    wins_window = 0
    window_size = 1000
    wr_history = []

    for hand in range(1, sarsa_hands + 1):
        r = train_one_hand_sarsa_bnn(env, agent, agent_id=0)
        total_reward_window += r
        if r > 0:
            wins_window += 1

        if hand % window_size == 0:
            elapsed = time.time() - start
            avg_r = total_reward_window / window_size
            wr = wins_window / window_size
            wr_history.append(wr)
            if hand % 5000 == 0:
                print(f"  {hand // 1000}k | Qsize={agent.get_q_table_size():>5} | "
                      f"AvgR={avg_r:+.2f} | WR={wr:.1%} | time={elapsed:.1f}s")
            total_reward_window = 0.0
            wins_window = 0

    # Save model
    model_path = os.path.join(out_dir, "nn_mc_vs_expert.pt")
    agent.save_model(model_path)
    total_time = time.time() - start

    # ---- Results summary ----
    last_5k_wr = np.mean(wr_history[-5:]) if len(wr_history) >= 5 else np.mean(wr_history)
    peak_wr = max(wr_history) if wr_history else 0.0

    print(f"\n  {'='*50}")
    print(f"  EXPERIMENT {exp_id} RESULTS:")
    print(f"    State mode:     {config['state_mode']}")
    print(f"    BNN arch:       {config['bnn_hidden_dims']}")
    print(f"    Q-table size:   {agent.get_q_table_size()}")
    print(f"    Last 5k avg WR: {last_5k_wr:.1%}")
    print(f"    Peak WR (1k):   {peak_wr:.1%}")
    print(f"    Train time:     {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  {'='*50}")

    return {
        'exp_id': exp_id,
        'q_table_size': agent.get_q_table_size(),
        'last_5k_wr': last_5k_wr,
        'peak_wr': peak_wr,
        'wr_history': wr_history,
        'time': total_time,
    }


def main():
    exp_ids = sys.argv[1].upper().split(',') if len(sys.argv) > 1 else ['A', 'B', 'C']
    sarsa_hands = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    bnn_data_hands = int(sys.argv[3]) if len(sys.argv) > 3 else 20000

    print("=" * 70)
    print("  NN_MC ABLATION EXPERIMENTS")
    print(f"  Experiments: {exp_ids}")
    print(f"  BNN data: {bnn_data_hands} hands, SARSA: {sarsa_hands} hands")
    print("=" * 70)

    results = []
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENT_CONFIGS:
            print(f"  WARNING: Unknown experiment '{exp_id}', skipping")
            continue
        result = run_experiment(
            exp_id,
            bnn_data_hands=bnn_data_hands,
            sarsa_hands=sarsa_hands,
        )
        results.append(result)

    # ---- Final comparison ----
    print("\n\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"  {'Exp':<5} {'Config':<35} {'Q-size':<8} {'Last5k WR':<12} {'Peak WR':<10} {'Time':<10}")
    print("-" * 80)
    for r in results:
        config = EXPERIMENT_CONFIGS[r['exp_id']]
        print(f"  {r['exp_id']:<5} {config['name']:<35} "
              f"{r['q_table_size']:<8} {r['last_5k_wr']:<12.1%} "
              f"{r['peak_wr']:<10.1%} {r['time']:<10.0f}s")
    print("-" * 80)
    print("  SARSA baseline reference: ~49% WR (784 states, 200k hands)")
    print("  Previous prob3 result:    ~38% WR (1072 states, 50k hands)")


if __name__ == "__main__":
    main()
