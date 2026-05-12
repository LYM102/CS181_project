# CS181_project
## Adapted Texas Hold'em Poker
---
### I. Game Setup
- **Players**: Fixed 2 players (AI vs AI)
- **Initial Chips**: 1000 per player
- **Blind Bets**:
  - Player 1 (Small Blind): Mandatory bet **5**
  - Player 2 (Big Blind): Mandatory bet **10**
- **Initial Pot**: 5 + 10 = 15
- **Raise Limit**: Maximum 3 continuous raising rounds, maximum bet **80**
  - Doubling sequence: 10 → 20 → 40 → 80
  - Once reaching 80, raising is prohibited; only Call or Fold is allowed

---
### II. Card Dealing & Community Cards Process
1. **Hole Cards**: Each player is dealt 2 private hole cards (only visible to themselves)
2. **Community Cards**: First deal 3 face-up community cards; the remaining community cards are revealed later.

> Note: Only 3 community cards are visible during the betting phase; there is only one betting round in total.

---
### III. Betting Phase (Only One Game Round)
- **Action Order**: Player 1 (Small Blind) decides first → Player 2 (Big Blind) decides next → Take actions alternately until termination conditions are met.
- **Three Valid Actions**:
  - **Fold**: Abandon the current round; all committed chips belong to the opponent, game ends immediately.
  - **Call**: Supplement own bet amount to match the current highest bet and continue the game.
  - **Raise**: Double the current highest bet following the sequence 10→20→40→80.
  Only available when current bet < 80 and total raising rounds < 3.
- **Raising Restrictions**:
  - Total raising rounds of both players ≤ 3
  Example: Player1 raises to 20 → Player2 raises to 40 → Player1 raises to 80, total 3 rounds.
  - After reaching 80, betting amount is locked and no more Raise is allowed.
- **Termination Conditions**:
  - Any player chooses **Fold** → The opponent wins and takes the whole pot.
  - Both players choose **Call** consecutively with no raising → Enter showdown to compare card strength.

---
### IV. Showdown and Settlement
- **Community Cards**: All 5 community cards are fully revealed at showdown for card combination.
- **Hand Ranking (From Highest to Lowest)**:
Straight Flush > Four of a Kind > Full House > Flush > Straight > High Card

- **Card Comparison Rule**: Each player selects the optimal 5-card combination from 2 hole cards + 5 community cards for comparison.
- **Result Settlement**:
  - The player with higher hand ranking wins the entire pot.
  - If hand rankings are equal, split the pot equally.

---
### V. State Space Definition
To simplify reinforcement learning modeling, game states are discretized as below:

| Dimension | Value Range | Description |
| :--- | :--- | :--- |
| Hand Strength Level | 0 ~ 19 | Score and log binning for hole cards + 3 community cards via `treys` library |
| Current Bet Level | 0,1,2,3 | Correspond to 10, 20, 40, 80 |
| Position | 0 or 1 | 0 = Player 1 (Small Blind), 1 = Player 2 (Big Blind) |

**Action Space**: 3 actions {Fold, Call, Raise}; Raise is invalid at the maximum bet level.
**Reward Function (for training)**:
- Reward equals the chip change of the player.

---
### VI. Supported AI Agents
This project implements the following agents for performance comparison:
1. **Random Agent** (baseline)
2. **Expectimax Agent**
3. **Monte Carlo (MC) Agent**
4. **SARSA Agent**
5. **DQN Agent**

---
### VII. Training and Evaluation Protocol
- **Training Mode**: Self-play or compete against fixed opponents (e.g., Random Agent).
- **Evaluation Metrics**:
  - Win rate of 5000 matches against Random Agent
  - Average net chip profit of 1000 matches in pairwise battles between different Agents
