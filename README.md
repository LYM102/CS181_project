# Standard Texas Hold'em — AI Training Platform (52-card)

Heads-up (2-player) Limit Texas Hold'em AI training and evaluation platform.
Standard 52-card deck with configurable betting limits.

---

## 1 Directory Structure

```
CS181_project/
├── main.py                     # Entry point (3 modes: interactive, evaluate, step)
├── requirements.txt            # Python dependencies
├── cfr_policy.pkl              # Pre-trained CFR strategy cache
├── README.md                   # This file
│
├── game/                       # ====== Core Game Module ======
│   ├── __init__.py
│   ├── constants.py            # Game constants (52 cards, blinds, betting levels, actions)
│   ├── card.py                 # Deck class + treys card utilities (52 cards)
│   ├── evaluator.py            # Hand evaluation, comparison, equity computation
│   ├── engine.py               # Game engine: deal → betting rounds → showdown
│   └── cfr_solver.py           # Custom CFR solver (External Sampling MCCFR)
│
├── agents/                     # ====== Agent Implementations ======
│   ├── __init__.py
│   ├── base_agent.py           # Abstract base class (unified interface)
│   ├── random_agent.py         # Random strategy Agent (baseline)
│   ├── expert_agent.py         # Expert Agent (CFR Nash equilibrium)
│   ├── sarsa_agent.py          # SARSA on-policy TD control Agent
│   ├── nn_mc_agent.py          # BNN (MC Dropout) + MC Q-table Agent
│   └── nfsp_agent.py           # Neural Fictitious Self-Play Agent (dual DNN)
│
└── train/                      # ====== Training Scripts ======
    ├── train_sarsa.py          # SARSA vs Expert training
    ├── train_nn_mc.py          # NN_MC two-phase training (BNN + Q-table)
    ├── train_nfsp.py           # NFSP self-play training
    ├── train_nn_mc_bluff.py    # BNN bluff detection experiment
    └── experiment_ablation.py   # Ablation study script
```

---

## 2 Game Rules

### 2.1 Basic Setup

| Parameter      | Value               | Description                              |
| -------------- | ------------------- | ---------------------------------------- |
| Game mode      | 2-player (heads-up) | Limit Texas Hold'em                      |
| Deck           | **52 cards**        | 4 suits (s/h/d/c) × 13 ranks (2~A)       |
| Starting chips | 1000                | Per player                               |
| Blinds         | SB=5, BB=10         | Heads-up: dealer = small blind           |
| Betting levels | {10,20,40,80,160,320} | 6-level betting                        |
| Max raises     | **4 per round**     | Highest bet = 320                        |

### 2.2 Action Space

$$\mathcal{A} = \{0: \text{Fold},\; 1: \text{Call},\; 2: \text{Raise}\}$$

- **Fold (0)**: Surrender the hand; opponent wins the pot
- **Call (1)**: Match the current bet level; acts as Check if no bet yet
- **Raise (2)**: Increase betting level by one; illegal when max level or raise limit reached

### 2.3 Game Flow

```
1. Post blinds (SB=5, BB=10)
2. Preflop — deal 2 hole cards each, SB acts first
3. Flop — deal 3 community cards, BB acts first
4. Turn — deal 1 community card
5. River — deal 1 community card (5 total)
6. Showdown — best 5-card hand wins
```

### 2.4 Hand Rankings (treys standard)

```
Royal/Straight Flush > Four of a Kind > Full House > Flush >
Straight > Three of a Kind > Two Pair > One Pair > High Card
```

### 2.5 Heads-up Special Rules

- Dealer = Small Blind, acts **first** preflop
- Non-dealer = Big Blind, acts **first** post-flop
- Dealer position alternates each hand

---

## 3 MDP Formulation

### 3.1 State Space

$$s = (H_{\text{code}},\; P_{\text{code}},\; B_{\text{level}},\; Pot_{\text{bin}},\; Pos)$$

| Component | Meaning         | Encoding                                                           |
| --------- | --------------- | ------------------------------------------------------------------ |
| H_code    | Own hand equity | Discretized into 20 bins (equity_to_bin)                           |
| P_code    | Community info  | Community card count (0/3/4/5)                                     |
| B_level   | Betting level   | {0, 1, 2, 3, 4, 5} → {10, 20, 40, 80, 160, 320}                   |
| Pot_bin   | Total pot size  | 7 bins: [0,30],(30,60],(60,120],(120,240],(240,480],(480,960],>960 |
| Pos       | Seat position   | {0, 1}                                                             |

**State space size**: 20 × 4 × 6 × 7 × 2 = **6720** (still tractable for tabular methods)

### 3.2 Reward Function

Sparse reward, given only at hand termination:

$$R = \begin{cases} P_{\text{pot}} & \text{win} \\ -B_{\text{own}} & \text{lose} \\ 0 & \text{tie} \end{cases}$$

### 3.3 Hyperparameters

$$\gamma = 0.95,\quad \alpha = 0.1,\quad \varepsilon \leftarrow \max(0.10,\; \varepsilon \cdot 0.9998)$$

---

## 4 Agent Summary

| Agent    | Core Method                                        | State Encoding                                                    | Key Innovation             |
| -------- | -------------------------------------------------- | ----------------------------------------------------------------- | -------------------------- |
| **Random** | Uniform random over legal actions                 | —                                                 | Testing baseline           |
| **Expert** | External Sampling MCCFR → Nash equilibrium         | Info set (hole_bucket, comm_bucket, round, bet_level, raises)      | CFR regret minimization    |
| **SARSA** | Q(s,a) on-policy TD update (actual next action)    | (H_code, P_code, B_level, Pot_bin, Pos)                           | Tabular TD control         |
| **NN_MC** | BNN (MC Dropout) predicts opponent → Q-table       | (H_bin, B_level, Pot_bin, O_NN)                                    | Bayesian uncertainty       |
| **NFSP** | Dual DNN self-play (DQN + Policy Network)          | 15-dim normalized feature vector                                   | End-to-end Nash convergence|

### 4.1 Agent Details

#### ExpertAgent (CFR)
- **Algorithm**: External Sampling MCCFR (Monte Carlo Counterfactual Regret Minimization)
- **Info set**: (player, hole_bucket×10, community_bucket×4, round×4, bet_level×7, raises×5)
- **Training**: ~90s for 10k iterations on 52-card deck
- **Performance**: 71% WR vs Random (tested at 10k iter)

#### SarsaAgent
- **Algorithm**: On-policy TD learning with ε-greedy exploration
- **Q-table**: Defaultdict with ~1000 states visited during training
- **Training**: vs ExpertAgent, ε decays from 1.0→0.10

#### NN_MCAgent (BNN + Q-table)
- **BNN**: 3-layer MLP (128→64→32) with MC Dropout, 47-dim input features
- **Feature vector**: own equity, board texture, action matrices (4×4), pot odds, SPR
- **Prediction target**: Opponent hand strength (weak/mid/strong), trained on Expert vs SARSA data
- **Q-table state**: Includes BNN opponent belief as state component

#### NFSPAgent (Neural Fictitious Self-Play)
- **Reference**: Heinrich & Silver (2016)
- **Architecture**: Two DNNs (128→128 each) — DQN (best response) + Policy Network (average strategy)
- **Training**: Self-play with η=0.1 anticipatory dynamics
- **State encoding**: 15-dim normalized vector (equity, pot odds, SPR, position, etc.)
- **Convergence**: Average policy network → approximate Nash equilibrium

---

## 5 Core Module Details

### 5.1 game/constants.py — Game Constants

Centralized parameters for 52-card Texas Hold'em. Single point of modification for rule changes:

```python
SUITS = ['s', 'h', 'd', 'c']                         # 4 suits
RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']  # 13 ranks
DECK_SIZE = 52
BETTING_LEVELS = [10, 20, 40, 80, 160, 320]           # 6 levels
MAX_RAISES = 4                                         # per round
```

### 5.2 game/card.py — Cards & Deck

Standard 52-card deck built on the `treys` library. treys uses integer encoding for cards; our `Deck` class handles shuffle + deal.

### 5.3 game/evaluator.py — Hand Evaluation

| Function                                 | Description                                          |
| ---------------------------------------- | ---------------------------------------------------- |
| `evaluate_hand(hole, community)`         | Returns (rank, class_str); lower rank = stronger     |
| `compare_hands(hole1, hole2, community)` | Returns (1/0/-1, winner_hand_class)                  |
| `compute_equity(hole, community, sim=100)`| Win rate via MC sampling (100 samples, cached)       |
| `equity_to_bin(equity, bins=20)`         | Discretize [0,1] equity to 20-bin index              |
| `pot_to_bin(pot)`                        | Discretize pot size into 7 bins (0~6)                |

Equity computation uses 100-sample Monte Carlo with `@lru_cache(maxsize=20000)`. For 52 cards, exact enumeration is infeasible.

### 5.4 game/engine.py — Game Engine

Two interfaces:
1. **Step-based** (RL training): `reset_hand()` → `step(action)`
2. **Hand-based** (evaluation): `run_hand()` / `run(num_hands)`

Key data structures: `PlayerState`, `Observation` (MDP state), `HandResult`.

### 5.5 game/cfr_solver.py — CFR Solver

External Sampling MCCFR for 52-card deck:
- Preflop equity cache: C(52,2)=1326 combinations, computed with 50 opponent samples × 8 board samples
- ~1.5 minutes warmup for full cache
- Info sets: ~322 distinct sets discovered by 10k iterations

---

## 6 Expert Agent (CFR) Implementation

### 6.1 Algorithm

**Counterfactual Regret Minimization (CFR)** iteratively minimizes "counterfactual regret" to approximate Nash equilibrium. Each iteration:
1. Compute strategy from cumulative regrets via regret matching
2. Traverse game tree with External Sampling (sample opponent actions + chance)
3. Accumulate regrets + average strategy

### 6.2 Information Set Design

```
info_key = (player, hole_bucket, community_bucket,
            betting_round, betting_level, raises_this_round)
```

Compact encoding using only betting state (level + raises), not full action history — sufficient for Limit Hold'em.

---

## 7 How to Run

### 7.1 Environment Setup

```bash
pip install -r requirements.txt   # treys + torch
```

### 7.2 Command Line

```bash
# Evaluate two agents
python main.py --mode evaluate --agent0 expert --agent1 random --num_hands 1000

# Interactive single hand
python main.py --mode interactive --num_hands 1

# Load trained models
python main.py --mode evaluate --agent0 sarsa --agent1 expert \
    --sarsa_model0 train/sarsa_model.pkl --num_hands 3000
```

### 7.3 Available Agent Types

| Key      | Class       | Description                       |
| -------- | ----------- | --------------------------------- |
| `random` | RandomAgent | Uniform random over legal actions |
| `expert` | ExpertAgent | CFR Nash equilibrium strategy     |
| `sarsa`  | SarsaAgent  | SARSA on-policy TD control        |
| `nn_mc`  | NN_MCAgent  | BNN (MC Dropout) + MC Q-table     |
| `nfsp`   | NFSPAgent   | Neural Fictitious Self-Play       |

### 7.4 Training

```bash
# SARSA vs Expert
python train/train_sarsa.py 200000 train/sarsa_model.pkl

# NN_MC (BNN + Q-table)
python train/train_nn_mc.py 50000 train/nn_mc_model.pt

# NFSP Self-Play
python -u train/train_nfsp.py 1000000 train/nfsp_model.pt
```

### 7.5 Python API

```python
from game.engine import GameEngine
from agents.expert_agent import ExpertAgent
from agents.random_agent import RandomAgent

engine = GameEngine(ExpertAgent(), RandomAgent())
results = engine.run(num_hands=500)

wins = sum(1 for r in results if r.winner == 0)
print(f"Expert WR: {wins / 500 * 100:.1f}%")
```

### 7.6 Training ExpertAgent from Scratch

```python
from agents.expert_agent import ExpertAgent

# Train CFR (first run ~2 min), auto-saved to cfr_policy.pkl
agent = ExpertAgent(train_iterations=50000)

# Subsequent loads from cache (instant)
agent = ExpertAgent()
```

---

## 8 Performance Benchmarks

| Matchup        | 52-card WR   | Notes                  |
| -------------- | :----------: | ---------------------- |
| Expert vs Random | **71%**    | 10k CFR iterations     |
| Random vs Random | ~50%      | Symmetric baseline     |

*More benchmarks pending full training runs on 52-card deck.*

---

## 9 Design Notes

### 9.1 Why Limit Hold'em

Discrete betting levels make the action space small (3 actions), enabling:
- Tabular methods (SARSA, MC) without function approximation
- CFR with compact info sets
- Focus on strategic depth rather than bet sizing

### 9.2 State Abstraction

Hole cards are abstracted via equity discretization (20 bins) rather than raw card values. This enables:
- Small Q-table (~6720 states) — tractable for tabular RL
- CFR info sets (~322 after 10k iterations)
- Cross-hand generalization

### 9.3 Equity Computation on 52-card Deck

MC sampling (100 samples) with LRU caching provides fast, approximate equity at each decision point. Cache hit rate >90% in practice.

---

## References

- Zinkevich et al. (2008): "Regret Minimization in Games with Incomplete Information"
- Heinrich & Silver (2016): "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Sutton & Barto: "Reinforcement Learning: An Introduction"
