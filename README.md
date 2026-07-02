# Belief-Gated Policies in Abstracted Texas Hold'em

A research codebase for *Belief-Gated Policies in Abstracted Texas Hold'em* — a four-stage agent ladder that progressively incorporates opponent modeling into decision-making: tabular SARSA → belief-augmented SARSA → residual belief-gated SARSA → neural policy transfer.

We use a standard 52-card deck, treys-based hand strength, fixed betting levels, and per-hand stack reset for evaluation.

---

## Agent Ladder

| Stage | Class | Policy | Opponent Modeling |
|-------|--------|--------|-------------------|
| **L0** | `SarsaAgent` | Tabular Q (SARSA) | None |
| **L1** | `L1Agent` | Same Q-table | BNN belief → state augmentation (τ = 0.65) |
| **L2** | `L2Agent` | Same Q-table as L1 | BNN + learned residual action gate |
| **L3** | `L3Agent` | CFR-distilled neural policy | Same belief + gate transferred to neural logits |

**Reference opponents:** `ExpertAgent` (tabular CFR), `AggressiveAgent` (exploitable bluff/trap patterns), `RandomAgent`.

### Core Modules

1. **SARSA Base Policy (L0)** — Tabular SARSA over a 5-tuple state abstraction `(hand_strength, community_cards, bet_level, pot_size, position)`. Trained against Random.

2. **Belief Module (L1)** — A BNN (MLP + MC Dropout) predicts opponent hand-strength class (weak/mid/strong) from a 53-dim feature vector. Belief is injected into the state only when confidence ≥ 0.65.

3. **Residual Gating Network (L2)** — A small MLP that adds a correction vector to the base action scores:
   \[
   \mathbf{z}' = \mathbf{z}_{\text{base}} + g_\theta(\mathbf{x})
   \]
   where \(\mathbf{x}\) includes base scores, belief probabilities, uncertainty, and betting-context features. Trained via supervised imitation of an exploit oracle.

4. **Neural Policy Transfer (L3)** — The same belief + gate framework applied to a CFR-distilled residual MLP policy, demonstrating that the gating mechanism generalizes beyond tabular representations.

---

## Project Structure

```
CS181_project/
├── main.py                     # Interactive / evaluate / step modes
├── eval_progressive.py         # Main ladder eval (L0–L3 vs Random/Aggressive/CFR)
├── retrain_and_eval.py         # Full retrain pipeline
├── requirements.txt
│
├── game/
│   ├── engine.py               # Game engine + Observation
│   ├── match_eval.py           # Per-hand stack reset, AvgR / win rate
│   ├── evaluator.py            # treys hand strength φ, bins
│   └── cfr_solver.py           # External-sampling MCCFR
│
├── agents/
│   ├── sarsa_agent.py          # L0: tabular SARSA
│   ├── l1_agent.py             # L1: belief-augmented SARSA
│   ├── l2_agent.py             # L2: + residual action gate
│   ├── l3_agent.py             # L3: neural policy + gate
│   ├── belief_features.py      # 53-dim feature encoder (shared)
│   ├── belief_net.py           # MC-Dropout BNN + training helpers
│   ├── belief_sarsa_agent.py   # Tabular Q + BNN base class
│   ├── belief_gating.py        # Learnable residual gate g_θ
│   ├── expert_agent.py         # CFR equilibrium opponent
│   ├── aggressive_agent.py     # Exploitative eval opponent
│   └── random_agent.py         # Random baseline
│
├── train/
│   ├── train_belief_net.py     # BNN opponent-strength classifier
│   ├── train_expert_distill.py # L3 neural policy (CFR distillation)
│   ├── train_gating_net.py     # Gate g_θ (L2 or L3 logits)
│   └── train_nn_mc_l1.py       # L1 gated-state SARSA Q-table
│
└── scripts/
    ├── collect_viz_data.py     # Collect per-decision-point data
    └── viz_experiments.py      # Generate paper figures
```

---

## Game Rules

| Parameter | Value |
|-----------|--------|
| Mode | 2-player heads-up |
| Deck | 52 cards |
| Starting stack | 1000 per hand (reset each hand in eval) |
| Blinds | SB = 5, BB = 10 |
| Betting levels | {10, 20, 40, 80, 160, 320} |
| Max raises / round | 4 |

Actions: Fold (0) / Call (1) / Raise (2).

---

## Quick Start

```bash
pip install -r requirements.txt
# Python 3.10+; PyTorch, numpy, treys
```

### Train (in order)

```bash
# 1. CFR solver + L0 SARSA
python retrain_and_eval.py --clean

# 2. BNN belief model
python -u train/train_belief_net.py

# 3. L3 neural policy (CFR distillation)
python -u train/train_expert_distill.py

# 4. Residual gating network
python -u train/train_gating_net.py

# 5. L1 Q-table
python -u train/train_nn_mc_l1.py
```

Or one shot: `python retrain_and_eval.py --clean --train-belief --train-distill --train-gating --train-l1`

### Evaluate

```bash
python eval_progressive.py --hands 1000 --seeds 42 43 44
```

### Head-to-head

```bash
python main.py --mode evaluate --agent0 l1 --agent1 aggressive --num_hands 1000
```

| CLI key | Agent |
|---------|--------|
| `random` | RandomAgent |
| `expert` | ExpertAgent (CFR) |
| `sarsa` | SarsaAgent (L0) |
| `l1` | L1Agent |
| `l2` | L2Agent |
| `l3` | L3Agent |

---

## Evaluation Metrics

- **AvgR** (primary): mean per-hand chip delta. Negative AvgR = agent loses chips on average.
- **WR** (secondary): fraction of hands won.

All evaluations use per-hand stack reset. Results are averaged over 3 random seeds × 1000 hands.

---

## References

- Zinkevich et al. (2008): CFR / regret minimization
- Gal & Ghahramani (2016): MC Dropout as approximate Bayesian inference
- Sutton & Barto: SARSA / TD learning
