# Minimalist Texas Hold'em

> YimingLi CunyuanZhang JiruiYu HaojiaZhang


## 1 Game Rules

### 1.1 Basic Settings

- **Game Mode**: Two-player AI vs AI  
- **Deck**: 16 custom cards (two suits, 7, 8, 9, 10, J, Q, K, A)  
- **Starting Chips**: 1000  
- **Blinds**: Small blind 5, big blind 10  
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

> We use the `treys` library for card representation and hand ranking.(*license:MIT*)



### 1.4 Rationale for Rule Simplification

The deck compression from 52 cards to 16 cards, together with the discrete betting cap, is **not an arbitrary trade-off** but a deliberate design driven by two computational concerns:

1. **Hand-strength evaluation cost**  
   The `treys`-based equity / hand-rank computation requires enumerating possible board completions. With a 52-card deck the enumeration grows combinatorially (e.g. $\binom{45}{2}$ remaining unknowns at the turn), making per-step strength evaluation prohibitively expensive inside a table-based RL training loop. Reducing the deck to 16 cards shrinks this enumeration by orders of magnitude.

2. **Bayesian posterior enumeration cost**  
   The Bayesian-MC Agent (Section 4.3) computes the posterior over opponent hand strength by enumerating all consistent opponent hole-card combinations conditioned on the visible cards. For a 52-card deck this enumeration is too large for online updates during self-play; the 16-card deck makes the marginalization tractable.

The resulting setting is essentially equivalent to the well-studied *Limit Texas Hold'em* variant and preserves the full structure of an imperfect-information game, so it remains a faithful testbed for the methods compared in this project.



## 2 MDP Mathematical Modeling

### 2.1 State Space

A state $s \in \mathcal{S}$ is composed of a 4-tuple, preserving raw card information (no manual compression):

$$
s = (H_{\text{code}},\;P_{\text{code}},\;B_{\text{level}},\;Pos)
$$

- $H_{\text{code}}$: player's hole cards equity code
- $P_{\text{code}}$: visible community cards  
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



## 4 Agent Implementation Logic

### 4.1 Expert Agent

This agent leverages the OpenSpiel library (by Google DeepMind) to provide a top-tier benchmark strategy. It approximates the Nash Equilibrium using the CFR (Counterfactual Regret Minimization) algorithm.


### 4.2 SARSA Agent

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



### 4.3 Bayesian-Monte Carlo Hybrid Agent
#### 4.3.1 Bayesian Inference Basic Formula

$$
P(H \mid A,B) = \frac{P(A \mid H,B)\cdot P(H)}{\sum_{i} P(A \mid H_i,B)\cdot P(H_i)}
$$
Where:
- $H\in\{H_\text{strong},H_\text{mid},H_\text{weak}\}$: discrete opponent hand strength level
- $A$: observable action taken by the opponent
- $B\in\{0,1,2,3\}$: current betting level
- $P(H)$: uniform prior probability of opponent hand strength at the start of each round
- $P(A\mid H,B)$: opponent action likelihood function under specific hand strength and betting level




#### 4.3.2 Data-Driven Likelihood Estimation
Abandon artificially customized fixed likelihood functions. This paper adopts a **two-stage interaction statistics method** to obtain real opponent behavioral likelihood:
1. **Likelihood pre-training stage**
Adopt a random policy agent to interact fully with the fixed heuristic opponent agent. Record the opponent’s real hand strength, current betting level and corresponding executed actions in each interaction step.
2. **Frequency-based probability calculation**
Count action occurrence frequency under grouped conditions, and apply Laplace smoothing to avoid zero probability:
$$
P(A\mid H,B)=\frac{N(H,B,A)+1}{N_\text{total}(H,B)+3}
$$



#### 4.3.3 State Space Definition For MC Method
Different from the original single-agent Monte Carlo method, the improved state incorporates the inferred opponent belief information:
$$
s = \big(S, B, O\big)
$$
- $S$: self-owned hand strength equity code
- $B$: current game betting level
- $O$: opponent belief label, determined by taking the category with the maximum posterior probability calculated by Bayesian inference

> We hope we can use the Bayesian inference result to guess the opponent’s hand strength, and then use the MC method to learn the optimal action.

### 4.4 NN-Monte Carlo Agent

#### 4.4.1 Motivation

The Bayesian‑MC agent (Section 4.3) only uses the opponent’s **current action** to update hand strength belief. This makes it incapable of detecting multi‑step patterns, such as a bluff that involves a check on the flop followed by a raise on the turn. Extending the discrete Bayesian model to a full Hidden Markov Model (HMM) is possible but cumbersome.

We therefore replace the Bayesian inference with a small **Bayesian Neural Network (BNN)** using MC Dropout. The BNN takes a **window of recent actions** (along with other observable features) and outputs a distribution over opponent hand strength. Its argmax provides the opponent label $O_{\text{NN}}$, which is then used in the same MC Q‑table structure.

#### 4.4.2 BNN Architecture & Input Features

**Motivation for round-wise structuring**: A flat sliding window over recent actions cannot tell whether a *Raise* happened **before or after a new community card was revealed**. Yet exactly this distinction is the key signal of multi-step bluffs (e.g. *flop check → turn raise*). We therefore organize action history into **per-round 2-D matrices**, so that pre-community and post-community actions are processed separately.

**Per-round action matrices**:

$$
M_{\text{opp}} \in \mathbb{R}^{R \times k},\qquad M_{\text{self}} \in \mathbb{R}^{R \times k}
$$

- $R = 4$: betting rounds (preflop / flop / turn / river)
- $k = 4$: max action slots per round (consistent with the 3-raise cap)
- Entry value: $0=\text{Fold},\;1=\text{Call},\;2=\text{Raise}$ rescaled to $[0,1]$; empty slots padded with $-1$

**Input features**:

| Feature                                     | Encoding                               |
| ------------------------------------------- | -------------------------------------- |
| Own hand equity                             | scalar in [0,1]                        |
| Community card strength                     | scalar in [0,1] (using `treys`)        |
| Betting round                               | one‑hot (4 dims)                       |
| **Opponent action matrix** $M_{\text{opp}}$ | shape $R \times k$, padded with $-1$   |
| **Own action matrix** $M_{\text{self}}$     | shape $R \times k$, padded with $-1$   |
| New‑community‑card flag                     | binary (1 if a new round just started) |

Each row of $M_{\text{opp}}$ / $M_{\text{self}}$ is first encoded by a small shared 1‑D module (mean‑pool or Conv1D) to obtain a per‑round behavioral summary; these summaries are then concatenated across rounds and combined with the scalar / one‑hot features before entering the BNN. This row‑wise design lets the network capture cross‑round patterns such as **flop check → turn raise**, the canonical bluff signature.

#### 4.4.3 Integration with Monte Carlo

State for MC Q‑table: $s = (S, B, O_{\text{NN}})$
- $S$: own equity discretized into 20 bins.
- $B$: betting level $\{0,1,2,3\}$.
- $O_{\text{NN}}$: argmax from BNN.