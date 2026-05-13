# Minimalist Texas Hold'em

## 1 Game Rules

### 1.1 Basic Settings

- **Game Mode**: Two-player AI vs AI  
- **Deck**: 16 custom cards (two suits, 7, 8, 9, 10, J, Q, K, A)  
- **Starting Chips**: 1000  
- **Blinds**: Small blind 5, big blind 10 → initial pot 15  
- **Betting Levels**: 10, 20, 40, 80 (max 3 raises, highest bet 80)

### 1.2 Procedure and Actions

- Each player is dealt 2 private hole cards; 3 community cards are revealed on the flop, and the showdown uses 5 community cards.  
- Legal action set:  
  $$
  \mathcal{A} = \{0:\text{Fold},\;1:\text{Call},\;2:\text{Raise}\}
  $$

### 1.3 Hand Rankings

$$
\text{Straight Flush} > \text{Four of a Kind} > \text{Full House} > \text{Flush} > \text{Straight} > \text{High Card}
$$

---

## 2 MDP Mathematical Modeling

### 2.1 State Space

A state $s \in \mathcal{S}$ is composed of a 4-tuple, preserving raw card information (no manual compression):

$$
s = (H_{\text{code}},\;P_{\text{code}},\;B_{\text{level}},\;Pos)
$$

- $H_{\text{code}}$: one-hot encoding of the player's hole cards  
- $P_{\text{code}}$: encoding of visible community cards  
- $B_{\text{level}} \in \{0,1,2,3\}$: betting level  
- $Pos \in \{0,1\}$: player position

### 2.2 Reward Function

Sparse reward is given only at the end of a hand:

$$
R =
\begin{cases}
P_{\text{sum}} & \text{Hand win} \\
-B_{\text{own}} & \text{Hand loss} \\
0 & \text{Tie}
\end{cases}
$$

- $P_{\text{sum}}$: total pot size  
- $B_{\text{own}}$: total amount bet by the player in this hand

### 2.3 Other Hyperparameters

$$
\gamma = 0.95,\qquad \alpha = 0.1,\qquad \varepsilon \leftarrow \max(0.01,\; \varepsilon \cdot 0.999)
$$

---

## 3 Basic Reinforcement Learning Formulas

### 3.1 Action-Value Function

$$
Q(s,a) = \mathbb{E}\left[G_t \mid S_t = s, A_t = a\right]
$$

### 3.2 Bellman Equation

$$
Q(s,a) = \mathbb{E}\left[R_{t+1} + \gamma Q(s',a')\right]
$$

### 3.3 $\varepsilon$-Greedy Policy

$$
A_t =
\begin{cases}
\text{Random}(\mathcal{A}) & \xi < \varepsilon \\
\arg\max_{a} Q(s,a) & \xi \ge \varepsilon
\end{cases},
\qquad \xi \sim U(0,1)
$$

---

## 4 Agent Implementation Logic

### 4.1 Heuristic Agent

Lookup table based on **hand strength + betting level**, fixed rules, no training.  
Hand strength is discretized into 10 levels (0–9). The decision rules are as follows:

| Hand Strength | Bet Level 0 (10) | Bet Level 1 (20) | Bet Level 2 (40) | Bet Level 3 (80) |
|---------------|------------------|------------------|------------------|------------------|
| 9 (very strong)| Raise            | Raise            | Raise            | Call             |
| 8             | Raise            | Raise            | Call             | Call             |
| 7             | Raise            | Call             | Call             | Call             |
| 6             | Call             | Call             | Call             | 50%Call/50%Fold  |
| 5             | Call             | Call             | Fold             | Fold             |
| 4             | Call             | Fold             | Fold             | Fold             |
| 3             | 80%Fold/20%Call  | Fold             | Fold             | Fold             |
| 2–0 (very weak)| Fold            | Fold             | Fold             | Fold             |

### 4.2 Monte Carlo Agent (MC)

#### Core Formulas

Discounted return:
$$
G_t = \sum_{k=t+1}^{T} \gamma^{\,k-t-1} R_k
$$

Incremental update:
$$
N(s,a) \leftarrow N(s,a) + 1,\qquad
Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\big(G_t - Q(s,a)\big)
$$

#### Implementation Logic

1. Record the full episode trajectory $(s,a)$  
2. After episode termination, compute $G_t$ backwards for each step  
3. Update Q-table using first-visit Monte Carlo  
4. Use $\varepsilon$-greedy during training, greedy during testing

### 4.3 SARSA Agent

#### Core Formulas

Temporal difference error:
$$
\delta_t = R + \gamma Q(s',a') - Q(s,a)
$$
Update rule:
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \delta_t
$$

#### Implementation Logic

Sample $(s,a,r,s',a')$ at each step, update Q-value online in real time, on-policy.

### 4.4 Pure Bayesian Agent

#### 4.4.1 Basic Formula

$$
P(H \mid A) = \frac{P(A \mid H)\,P(H)}{P(A)},\quad
P(A) = \sum_{i} P(A \mid H_i)\,P(H_i)
$$

- $H$: opponent's hidden hand  
- $A$: observed action  
- $P(H)$: uniform prior $\displaystyle P(H_i) = \frac{1}{\binom{n}{2}}$  
- $P(A \mid H)$: conditional likelihood function

#### 4.4.2 Custom Likelihood Function (Improved)

Combines opponent hand strength and current betting level $B \in \{0,1,2,3\}$:

$$
P(A \mid H, B) =
\begin{cases}
P(\text{Raise}\mid H_{\text{strong}},B) = 0.75 - 0.15B \\
P(\text{Call}\mid H_{\text{strong}},B) = 0.20 + 0.10B \\
P(\text{Fold}\mid H_{\text{strong}},B) = 0.05 + 0.05B \\
P(\text{Raise}\mid H_{\text{mid}},B) = 0.40 - 0.10B \\
P(\text{Call}\mid H_{\text{mid}},B) = 0.50 \\
P(\text{Fold}\mid H_{\text{mid}},B) = 0.10 + 0.10B \\
P(\text{Raise}\mid H_{\text{weak}},B) = 0.15 - 0.05B \\
P(\text{Call}\mid H_{\text{weak}},B) = 0.25 \\
P(\text{Fold}\mid H_{\text{weak}},B) = 0.60 + 0.05B
\end{cases}
$$

> Design rationale: Higher betting levels make strong hands more conservative (lower raise probability) and weak hands more likely to fold, aligning with poker risk management.

#### 4.4.3 Decision Formula

Expected utility:
$$
U(a) = \sum_{H} P(H \mid A) \cdot R(s,a,H)
$$
Optimal action:
$$
a^* = \arg\max_{a \in \mathcal{A}} U(a)
$$

### 4.5 Bayesian + Monte Carlo Agent

#### 4.5.1 Core Logic
Keep all MC update formulas, only modify the sampling distribution:
- **Vanilla MC**: uniform sampling of opponent hands  
- **Improved MC**: weighted sampling according to Bayesian posterior $P(H \mid A)$

#### 4.5.2 Execution Flow

1. Observe opponent action, compute posterior distribution over opponent hands  
2. Sample opponent hands according to the posterior for simulated playouts  
3. Generate trajectories and compute discounted returns $G_t$  
4. Update Q-table using MC incremental average formulas

> Author YiMingLi
