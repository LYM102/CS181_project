# Texas Hold'em AI — Multi-Agent Training & Evaluation Platform

A heads-up (2-player) Limit Texas Hold'em AI training and evaluation platform.
Standard 52-card deck with fixed betting levels.

---

## 1 Project Structure

```
CS181_project/
├── main.py                          # Entry point: 3 modes (interactive/evaluate/step)
├── requirements.txt                 # Python dependencies
├── cfr_policy.pkl                   # Pre-trained CFR strategy cache
│
├── game/                            # ====== Core Game Module ======
│   ├── constants.py                 # Game constants (52-card deck, blinds, betting levels)
│   ├── card.py                      # Deck + treys card utilities
│   ├── evaluator.py                 # Hand evaluation, comparison, equity computation
│   ├── engine.py                    # Game engine: deal → betting rounds → showdown
│   └── cfr_solver.py               # External Sampling MCCFR solver
│
├── agents/                          # ====== Agent Implementations ======
│   ├── base_agent.py                # Abstract base class (unified interface)
│   ├── random_agent.py              # Random baseline
│   ├── expert_agent.py              # CFR Nash equilibrium Expert
│   ├── sarsa_agent.py               # SARSA on-policy TD learning
│   ├── nn_mc_agent.py               # BNN-Policy + NN_MC Agent (core)
│   ├── nfsp_agent.py                # Neural Fictitious Self-Play (dual DNN)
│   └── aggressive_agent.py          # Aggressive/tight-passive agents (training diversity)
│
├── train/                           # ====== Training Scripts ======
│   ├── compare_all.py               # Three-way comparison: BNN vs SARSA vs Expert
│   ├── train_bnn_policy.py          # BNN-Policy four-phase training
│   ├── train_expert_distill.py      # Expert policy distillation
│   ├── run_phase3_phase4.py         # Online RL + Self-play training
│   └── results/policy/              # Training artifacts
│       ├── expert_distill_sp_sp.pt  # ★ Best BNN model
│       └── sarsa_trained.pkl        # ★ Trained SARSA Q-table
│
└── logs/
    └── compare_all.out              # ★ Final three-way comparison results
```

---

## 2 Game Rules

| Parameter      | Value                        | Description                       |
| -------------- | ---------------------------- | --------------------------------- |
| Mode           | 2-player heads-up            | Limit Texas Hold'em               |
| Deck           | **52 cards**                 | 4 suits × 13 ranks               |
| Starting chips | 1000                         | Per player                        |
| Blinds         | SB=5, BB=10                  | heads-up: dealer = small blind    |
| Betting levels | {10, 20, 40, 80, 160, 320}  | 6 levels                          |
| Max raises     | **4 per round**              |                                   |

Action space: \(\mathcal{A} = \{\text{Fold}, \text{Call}, \text{Raise}\}\)

---

## 3 Agent Summary

| Agent | Core Method | Performance Tier |
|-------|------------|-----------------|
| **Random** | Uniform over legal actions | Baseline |
| **Expert** | External Sampling MCCFR → Nash equilibrium | Strong baseline |
| **SARSA** | Q-table + on-policy TD learning (6720-dim state) | ≈ Expert |
| **NN_MC** | BNN (MC Dropout) + Q-table hybrid | Classical approach |
| **NFSP** | Dual DNN self-play (DQN + average policy net) | End-to-end Nash |
| **BNN-Policy** | BNN policy net + four-phase training pipeline | ★ Current best |

---

## 4 BNN-Policy Architecture

### 4.1 Network Structure

```
Input (53-dim features)
    │
    ├─ ResidualBlock(53 → 256)
    │   ├─ Linear + LayerNorm + ReLU
    │   ├─ Dropout(0.15)
    │   ├─ Linear(256→256) + LayerNorm
    │   └─ + Residual connection
    │
    ├─ ResidualBlock(256 → 128)   Same structure
    ├─ ResidualBlock(128 → 64)    Same structure
    │
    └─ Linear(64 → 3) → [Fold, Call, Raise] logits
```

### 4.2 53-Dimensional Input Features

| Index | Feature | Description |
|-------|---------|-------------|
| [0] | Own equity | [0,1] |
| [1:5] | Board texture (paired/flush-draw/straight-draw/connectivity) | 5 dims |
| [5:9] | Current round onehot | 4 dims |
| [9:25] | Opponent action history matrix (4 rounds × 4 action frequencies) | 16 dims |
| [25:41] | Self action history matrix | 16 dims |
| [41] | New community card flag | binary |
| [42] | Pot odds | [0,1] |
| [43] | SPR (stack-to-pot ratio) | [0,1] |
| [44:47] | Opponent hand strength (true value at training / 0.5 mask at inference) | 3 dims |
| [47:53] | Betting level / position / raises remaining / eff. stack / legal actions | 6 dims |

### 4.3 MC Dropout (Bayesian Uncertainty)

Dropout remains **active during inference**. 20 stochastic forward passes produce a distribution:

```
20 samples → mean μ(action) + variance σ²(action)
```

- Low variance → the model is confident in its decision
- High variance → high uncertainty → favors conservative actions

### 4.4 Four-Phase Training Pipeline

```
Phase 1: Behavior Cloning  ──→  Cold-start from SARSA/Expert demonstrations
Phase 2: DAgger             ──→  Online policy distillation with SARSA oracle
Phase 3: REINFORCE + EWC    ──→  Online RL improvement + anti-forgetting
Phase 4: Self-play          ──→  Play against historical policy snapshots
```

- **DAgger**: SARSA acts as a live oracle, providing corrective labels for states the BNN actually visits, reducing distribution shift.
- **EWC** (Elastic Weight Consolidation): Penalizes deviations from important parameters learned in earlier phases, preventing catastrophic forgetting during RL fine-tuning.

---

## 5 Final Results

**Three-way comparison**: BNN-SP2 vs SARSA vs Expert, 3000 hands each.

| Matchup | Win Rate | AvgR Difference |
|---------|----------|----------------|
| BNN-SP2 vs Expert | 48.1% vs 48.2% | **BNN +1.25 chips/hand** |
| BNN-SP2 vs SARSA | 83.7% vs 14.9% | **BNN +5.04 chips/hand** |
| SARSA vs Expert | 48.5% vs 47.6% | **SARSA +2.94 chips/hand** |

> **Note on AvgR**: `chips_before` is captured after blinds are posted, so AvgR sums are offset by the blinds (≈15 chips/hand). The **difference** between agents' AvgR values represents the true skill gap in this zero-sum game.

### Performance Ranking

```
BNN-SP2  ⪆  SARSA  ⪆  Expert  >>>>  Random
```

All three top agents achieve near-Nash-equilibrium performance. BNN-Policy holds a slight edge in chip efficiency.

### SARSA Improvement

The original SARSA training had a critical bug: the final action of each hand never received a terminal reward update, resulting in only **9.1%** win rate vs Expert. After fixing the bug (+ epsilon decay + doubled training hands), the Q-table grew from 117 entries to 676, and win rate jumped to **48.5%**.

---

## 6 Quick Start

### 6.1 Environment

```bash
pip install -r requirements.txt   # treys + torch + numpy
```

### 6.2 Evaluation

```bash
# Three-way comparison (auto-trains SARSA if needed)
python -u train/compare_all.py

# Custom head-to-head evaluation
python main.py --mode evaluate --agent0 expert --agent1 sarsa \
    --sarsa_model0 train/results/policy/sarsa_trained.pkl --num_hands 3000
```

### 6.3 Available Agent Types

| Key | Agent |
|-----|-------|
| `random` | Random baseline |
| `expert` | CFR Nash equilibrium |
| `sarsa` | SARSA Q-table |
| `nn_mc` | BNN + Q-table hybrid |
| `nfsp` | Neural Fictitious Self-Play |

---

## 7 Design Notes

### Why Limit Hold'em

Discrete betting levels constrain the action space to {Fold, Call, Raise}, which enables:
- Tabular methods (SARSA) to train on a 6720-dimensional state space
- Compact CFR information sets (~322 sets)
- Focus on strategic depth rather than bet sizing optimization

### State Abstraction

Hole cards are encoded via equity discretization (20 bins) rather than raw card values, combined with round, betting level, pot size, and position to form a 6720-dimensional state space. BNN-Policy uses a richer 53-dimensional feature vector including action history matrices, board texture, pot odds, and SPR.

### References

- Zinkevich et al. (2008): "Regret Minimization in Games with Incomplete Information"
- Heinrich & Silver (2016): "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Ross et al. (2011): "A Reduction of Imitation Learning to No-Regret Online Learning" (DAgger)
- Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in neural networks" (EWC)
