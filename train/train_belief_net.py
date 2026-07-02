"""Train BNN opponent-strength classifier."""
from __future__ import annotations

import sys
import os
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from game.engine import GameEngine
from agents.belief_net import (
    BNNWithMCDropout,
    collect_bnn_training_data,
    train_bnn,
    _equity_to_strength_label,
    _equity_to_strength_label_5class,
)
from agents.expert_agent import ExpertAgent
from agents.aggressive_agent import AggressiveAgent
from agents.random_agent import RandomAgent


BELIEF_MODEL = "train/results/policy/belief_net_v4.pt"


def collect_mixed_belief_data(num_hands: int = 50000,
                              mask_prob: float = 1.0,
                              verbose: bool = True,
                              num_classes: int = 3) -> tuple:
    """Collect opponent-strength labels from mixed matchups.

    Observer is Random (neutral); target is Expert/Aggressive.
    mask_prob=1.0 masks opponent features to match inference conditions.
    """
    import agents.belief_net as belief_net
    if num_classes == 5:
        original_label_fn = belief_net._equity_to_strength_label
        belief_net._equity_to_strength_label = _equity_to_strength_label_5class

    X_parts, y_parts, mask_parts = [], [], []
    forty_pct = int(num_hands * 0.4)
    twenty_pct = num_hands - 2 * forty_pct

    configs = [
        ("Aggressive", forty_pct, lambda: GameEngine(
            RandomAgent(name="Observer"), AggressiveAgent(name="Aggressive"))),
        ("Expert", forty_pct, lambda: GameEngine(
            RandomAgent(name="Observer"), ExpertAgent(name="Expert"))),
        ("Expert-vs-Aggressive", twenty_pct, lambda: GameEngine(
            ExpertAgent(name="ExpertObs"), AggressiveAgent(name="Aggressive"))),
    ]

    for opp_name, n_hands, make_env in configs:
        if verbose:
            print(f"\n  Collecting vs {opp_name} as TARGET ({n_hands} hands)...")
        env = make_env()
        X, y, mask_flags = collect_bnn_training_data(
            env, num_hands=n_hands, mask_prob=mask_prob,
            verbose=verbose, target_player=1)
        X_parts.append(X)
        y_parts.append(y)
        mask_parts.append(mask_flags)
        if verbose:
            print(f"    -> {len(X)} samples from {opp_name}")

    if num_classes == 5:
        belief_net._equity_to_strength_label = original_label_fn

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    mask_all = np.concatenate(mask_parts, axis=0)
    return X_all, y_all, mask_all


def train_belief_net(num_hands: int = 50000,
                     epochs: int = 200,
                     batch_size: int = 128,
                     lr: float = 1e-3,
                     mask_prob: float = 1.0,
                     num_classes: int = 3,
                     save_path: str = BELIEF_MODEL,
                     device: str = None,
                     use_label_smoothing: bool = True,
                     use_cosine_schedule: bool = True) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"  BNN Opponent Belief Model Training v4 ({num_classes}-class)")
    print(f"  Hands: {num_hands}  |  Epochs: {epochs}  |  Device: {device}")
    print(f"  Label smoothing: {use_label_smoothing}  |  Cosine LR: {use_cosine_schedule}")
    print("=" * 60)

    start = time.time()
    X, y, mask_flags = collect_mixed_belief_data(
        num_hands=num_hands, mask_prob=mask_prob, verbose=True,
        num_classes=num_classes)
    if num_classes == 3:
        print(f"\n  Total samples: {len(X)}  "
              f"(weak={np.sum(y==0)}, mid={np.sum(y==1)}, strong={np.sum(y==2)})")
    else:
        labels = ["v_weak", "weak", "mid", "strong", "v_strong"]
        dist = " ".join(f"{l}={np.sum(y==i)}" for i, l in enumerate(labels))
        print(f"\n  Total samples: {len(X)}  ({dist})")

    model = BNNWithMCDropout(
        input_dim=53, hidden_dims=(256, 128, 64),
        num_classes=num_classes, dropout_rate=0.1,
    ).to(device)

    model = train_bnn(
        model, X, y, mask_flags=mask_flags,
        epochs=epochs, batch_size=batch_size, lr=lr,
        device=device, verbose=True,
        use_label_smoothing=use_label_smoothing,
        use_cosine_schedule=use_cosine_schedule)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "belief_net_state_dict": model.state_dict(),
        "belief_trained": True,
        "bnn_state_dict": model.state_dict(),  # alias for L1 checkpoint compat
        "bnn_trained": True,
    }, save_path)

    elapsed = time.time() - start
    print(f"\n  Belief model saved to {save_path} ({elapsed:.0f}s)")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train BNN opponent belief model")
    parser.add_argument("--hands", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", type=str, default=BELIEF_MODEL)
    parser.add_argument("--num-classes", type=int, default=3, choices=[3, 5])
    args = parser.parse_args()
    train_belief_net(num_hands=args.hands, epochs=args.epochs,
                     num_classes=args.num_classes, save_path=args.output)


if __name__ == "__main__":
    main()
