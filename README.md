# Texas Hold'em AI — Multi-Agent Training & Evaluation Platform

A heads-up (2-player) Limit Texas Hold'em AI platform with multiple agent implementations.
Standard 52-card deck, fixed betting levels.

---

## 1 Project Structure

```
CS181_project/
├── main.py                     # Entry point (interactive / evaluate / step modes)
├── requirements.txt            # Python dependencies
├── cfr_policy.pkl              # Pre-trained CFR strategy cache
│
├── game/                       # Core game module
│   ├── constants.py            # Game constants (blinds, betting levels, actions)
│   ├── card.py                 # 52-card deck + treys utilities
│   ├── evaluator.py            # Hand evaluation, comparison, equity computation
│   ├── engine.py               # Game engine: deal → betting rounds → showdown
│   └── cfr_solver.py           # External Sampling MCCFR solver
│
├── agents/                     # Agent implementations
│   ├── base_agent.py           # Abstract base class
│   ├── random_agent.py         # Random baseline
│   ├── expert_agent.py         # Expert (CFR Nash equilibrium)
│   ├── sarsa_agent.py          # SARSA (on-policy TD learning)
│   ├── nn_mc_agent.py          # BNN-Policy + NN_MC Agent
│   ├── nfsp_agent.py           # NFSP (Neural Fictitious Self-Play)
│   └── aggressive_agent.py     # Aggressive/Tight-Passive agents (training diversity)
│
├── train/                      # Training scripts
│   ├── compare_all.py          # Three-way comparison
│   ├── train_bnn_policy.py     # BNN-Policy training
│   ├── train_expert_distill.py # Expert policy distillation
│   └── run_phase3_phase4.py    # Online RL + Self-play
│
└── logs/
    └── compare_all.out         # Final comparison results
```

---

## 2 Game Rules

| Parameter      | Value                        |
| -------------- | ---------------------------- |
| Mode           | 2-player heads-up            |
| Deck           | 52 cards (4 suits × 13 ranks) |
| Starting chips | 1000 per player              |
| Blinds         | SB=5, BB=10                  |
| Betting levels | {10, 20, 40, 80, 160, 320}  |
| Max raises     | 4 per round                  |

Action space: Fold / Call / Raise.

---

## 3 Agents

### 3.1 RandomAgent

Uniform random selection over legal actions. Serves as the baseline.

### 3.2 ExpertAgent (CFR)

Uses External Sampling MCCFR to approximate a Nash equilibrium strategy. The information set is encoded as `(player, hole_bucket, community_bucket, round, betting_level, raises)`. The CFR solver produces ~322 distinct information sets with a mixed strategy over {Fold, Call, Raise} for each.

### 3.3 SarsaAgent

On-policy TD learning with a Q-table. The state is a 5-tuple `(equity_bin, community_card_count, betting_level, pot_bin, position)` with a state space of 20 × 4 × 6 × 7 × 2 = 6720. Uses ε-greedy exploration with decay.

### 3.4 NN_MCAgent

A hybrid agent combining a BNN (Bayesian Neural Network via MC Dropout) with a Q-table. The BNN predicts opponent hand strength (weak/mid/strong) from board texture and action history. The opponent belief is included as a state component in the Q-table.

### 3.5 NFSPAgent

Neural Fictitious Self-Play with two DNNs: a DQN (best-response network) and an average-policy network. Trained via self-play with anticipatory dynamics (η = 0.1). Uses a 15-dimensional normalized feature vector.

### 3.6 BNN-PolicyAgent ★

An end-to-end neural policy network that maps 53-dimensional features directly to action logits {Fold, Call, Raise}.

**Network Architecture:**

```
Input (53-dim features)
    │
    ├─ ResidualBlock(53 → 256)
    │   ├─ Linear + LayerNorm + ReLU
    │   ├─ Dropout(0.15)
    │   ├─ Linear(256→256) + LayerNorm
    │   └─ + Residual connection
    │
    ├─ ResidualBlock(256 → 128)
    ├─ ResidualBlock(128 → 64)
    │
    └─ Linear(64 → 3) → action logits
```

**53-Dimensional Input Features:**

| Index | Feature | Description |
|-------|---------|-------------|
| [0] | Own equity | [0,1] |
| [1:5] | Board texture | Paired/flush-draw/straight-draw/connectivity (5 dims) |
| [5:9] | Current round onehot | Preflop/Flop/Turn/River (4 dims) |
| [9:25] | Opponent action history | 4 rounds × action frequencies (16 dims) |
| [25:41] | Self action history | Same structure (16 dims) |
| [41] | New community card flag | Binary |
| [42] | Pot odds | [0,1] |
| [43] | SPR (stack-to-pot ratio) | [0,1] |
| [44:47] | Opponent hand strength | True value at training, 0.5 mask at inference (3 dims) |
| [47:53] | Betting level / position / raises remaining / eff. stack / legal actions | 6 dims |

**MC Dropout:** Dropout remains active during inference. 20 stochastic forward passes produce a mean action distribution and an uncertainty estimate (variance). High uncertainty leads to more conservative action selection.

**Training Pipeline:**

| Phase | Method | Purpose |
|-------|--------|---------|
| 1 | Behavior Cloning | Supervised pretraining on SARSA/Expert demonstrations |
| 2 | DAgger | Online policy distillation with SARSA as oracle |
| 3 | REINFORCE + EWC | Online RL with anti-forgetting regularization |
| 4 | Self-play | Play against historical policy snapshots |

---

## 4 Results

### 4.1 Evaluation Metrics

Each matchup runs 3000 hands. Two metrics are reported:

- **Win Rate (WR)**: fraction of hands won.
- **AvgR (Average Reward)**: `Δchips / num_hands`, where `Δchips = chips_after_hand − chips_before_hand`. `chips_before` is captured after blinds are posted, so the blind amount (SB+BB=15 chips) is already in the pot — hence **both agents' AvgR values are positive**. The **difference** between them measures the true skill gap.

### 4.2 Three-Way Comparison

| Matchup | Win Rate | AvgR Difference |
|---------|----------|----------------|
| BNN-SP2 vs Expert | 48.1% / 48.2% | BNN +1.25 chips/hand |
| BNN-SP2 vs SARSA | 83.7% / 14.9% | BNN +5.04 chips/hand |
| SARSA vs Expert | 48.5% / 47.6% | SARSA +2.94 chips/hand |

### 4.3 Performance Ranking

```
BNN-SP2  ⪆  SARSA  ⪆  Expert  >>>>  Random
```

---

## 5 Quick Start

```bash
pip install -r requirements.txt   # treys + torch + numpy

# Three-way comparison
python -u train/compare_all.py

# Custom head-to-head
python main.py --mode evaluate --agent0 expert --agent1 sarsa \
    --sarsa_model0 train/results/policy/sarsa_trained.pkl --num_hands 3000
```

| Agent Key | Implementation |
|-----------|---------------|
| `random` | RandomAgent |
| `expert` | ExpertAgent (CFR) |
| `sarsa` | SarsaAgent |
| `nn_mc` | NN_MCAgent |
| `nfsp` | NFSPAgent |

---

## References

- Zinkevich et al. (2008): "Regret Minimization in Games with Incomplete Information"
- Heinrich & Silver (2016): "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Ross et al. (2011): "A Reduction of Imitation Learning to No-Regret Online Learning" (DAgger)
- Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in neural networks" (EWC)
