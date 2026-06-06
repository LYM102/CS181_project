# agents/bayesian_model.py
"""
Bayesian opponent model for Texas Hold'em.

This module contains two components:

1. BayesianLikelihoodEstimator
   - Collects frequency counts of opponent behavior.
   - Estimates P(action | hand_strength_bucket, betting_level)
     with Laplace smoothing.

2. BayesianOpponentModel
   - Maintains posterior belief over opponent hand strength.
   - Updates P(hand_strength | observed_action, betting_level)
     after observing opponent actions.

Strength buckets:
    0 = weak
    1 = mid
    2 = strong

Actions follow game.constants:
    0 = Fold
    1 = Call
    2 = Raise
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Dict, Tuple, List

from game.constants import FOLD, CALL, RAISE, BETTING_LEVELS


# =========================
# Strength bucket constants
# =========================

WEAK = 0
MID = 1
STRONG = 2

NUM_STRENGTH_BUCKETS = 3
NUM_ACTIONS = 3

BUCKET_NAMES = {
    WEAK: "weak",
    MID: "mid",
    STRONG: "strong",
}

ACTION_NAMES_LOCAL = {
    FOLD: "fold",
    CALL: "call",
    RAISE: "raise",
}


def equity_to_strength_bucket(equity: float) -> int:
    """
    Convert hand equity into a discrete strength bucket.

    Thresholds are aligned with AggressiveAgent:
        equity < 0.30       -> weak
        0.30 <= equity < 0.60 -> mid
        equity >= 0.60      -> strong

    Args:
        equity: float in [0, 1]

    Returns:
        bucket: 0 weak, 1 mid, 2 strong
    """
    if equity < 0.30:
        return WEAK
    elif equity < 0.60:
        return MID
    else:
        return STRONG


def strength_bucket_name(bucket: int) -> str:
    """Return human-readable bucket name."""
    return BUCKET_NAMES.get(bucket, f"unknown({bucket})")


def normalize_bet_level(bet_level: int) -> int:
    """
    Normalize betting level into valid range.

    In the engine, raw self.betting_level may be -1 when no one has bet
    in a post-flop round. Observation already maps it to max(0, level),
    but this helper makes the Bayesian model robust.

    Args:
        bet_level: raw or observation betting level

    Returns:
        int in [0, len(BETTING_LEVELS)-1]
    """
    if bet_level is None:
        return 0
    if bet_level < 0:
        return 0
    return min(int(bet_level), len(BETTING_LEVELS) - 1)


def hand_bucket_from_cards(hole_cards: list[int],
                           community_cards: list[int],
                           sim: int = 50) -> int:
    """
    Compute strength bucket from actual cards.

    This is mainly used during likelihood pretraining, where we are allowed
    to access the true opponent hole cards.

    Args:
        hole_cards: player's private cards
        community_cards: visible board cards
        sim: Monte Carlo samples for equity computation

    Returns:
        bucket: 0 weak, 1 mid, 2 strong
    """
    from game.evaluator import compute_equity

    equity = compute_equity(hole_cards, community_cards, sim=sim)
    return equity_to_strength_bucket(equity)


# =================================
# Likelihood estimator: P(A | H, B)
# =================================

class BayesianLikelihoodEstimator:
    """
    Frequency-based estimator for opponent behavior likelihood.

    It estimates:

        P(action | hand_strength_bucket, betting_level)

    using Laplace smoothing:

        P(A | H, B) = (N(H,B,A) + alpha)
                      / (N(H,B) + alpha * num_actions)
    """

    def __init__(self,
                 num_buckets: int = NUM_STRENGTH_BUCKETS,
                 num_actions: int = NUM_ACTIONS,
                 num_bet_levels: int = None,
                 laplace: float = 1.0):
        self.num_buckets = num_buckets
        self.num_actions = num_actions
        self.num_bet_levels = num_bet_levels or len(BETTING_LEVELS)
        self.laplace = float(laplace)

        # counts[(bucket, bet_level, action)] = count
        self.counts: Dict[Tuple[int, int, int], int] = defaultdict(int)

        # totals[(bucket, bet_level)] = total actions observed
        self.totals: Dict[Tuple[int, int], int] = defaultdict(int)

    def record(self, hand_bucket: int, bet_level: int, action: int) -> None:
        """
        Record one observed action under known hand strength and bet level.

        Args:
            hand_bucket: 0 weak, 1 mid, 2 strong
            bet_level: betting level
            action: 0 fold, 1 call, 2 raise
        """
        if hand_bucket < 0 or hand_bucket >= self.num_buckets:
            raise ValueError(f"Invalid hand_bucket: {hand_bucket}")

        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}")

        b = normalize_bet_level(bet_level)

        self.counts[(hand_bucket, b, action)] += 1
        self.totals[(hand_bucket, b)] += 1

    def prob(self, action: int, hand_bucket: int, bet_level: int) -> float:
        """
        Return smoothed probability P(action | hand_bucket, bet_level).
        """
        if hand_bucket < 0 or hand_bucket >= self.num_buckets:
            raise ValueError(f"Invalid hand_bucket: {hand_bucket}")

        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}")

        b = normalize_bet_level(bet_level)

        count = self.counts[(hand_bucket, b, action)]
        total = self.totals[(hand_bucket, b)]

        numerator = count + self.laplace
        denominator = total + self.laplace * self.num_actions

        return numerator / denominator

    def action_distribution(self, hand_bucket: int, bet_level: int) -> List[float]:
        """
        Return [P(fold), P(call), P(raise)] for a given bucket and bet level.
        """
        return [
            self.prob(action, hand_bucket, bet_level)
            for action in range(self.num_actions)
        ]

    def total_records(self) -> int:
        """Return total number of recorded observations."""
        return sum(self.totals.values())

    def summary(self) -> str:
        """
        Return human-readable summary of likelihood table.
        Useful for debugging after pretraining.
        """
        lines = []
        lines.append("BayesianLikelihoodEstimator Summary")
        lines.append(f"Total records: {self.total_records()}")
        lines.append(f"Laplace alpha: {self.laplace}")
        lines.append("")

        for h in range(self.num_buckets):
            h_name = strength_bucket_name(h)
            for b in range(self.num_bet_levels):
                total = self.totals[(h, b)]
                dist = self.action_distribution(h, b)
                lines.append(
                    f"H={h_name:<6} B={b} total={total:<5} "
                    f"P(F)={dist[0]:.3f} P(C)={dist[1]:.3f} P(R)={dist[2]:.3f}"
                )

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """
        Save estimator to pickle file.
        """
        data = {
            "num_buckets": self.num_buckets,
            "num_actions": self.num_actions,
            "num_bet_levels": self.num_bet_levels,
            "laplace": self.laplace,
            "counts": dict(self.counts),
            "totals": dict(self.totals),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BayesianLikelihoodEstimator":
        """
        Load estimator from pickle file.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            num_buckets=data["num_buckets"],
            num_actions=data["num_actions"],
            num_bet_levels=data["num_bet_levels"],
            laplace=data["laplace"],
        )

        obj.counts.update(data["counts"])
        obj.totals.update(data["totals"])

        return obj


# =========================================
# Posterior model: P(H | observed actions)
# =========================================

class BayesianOpponentModel:
    """
    Maintains posterior belief over opponent hand strength.

    At the start of each hand:

        P(H) = uniform over weak/mid/strong

    After observing opponent action A at betting level B:

        P(H | A, B) ∝ P(A | H, B) * P(H)

    Sequential update is used, so multiple observed opponent actions in
    one hand accumulate evidence.
    """

    def __init__(self,
                 likelihood: BayesianLikelihoodEstimator,
                 prior: List[float] | None = None):
        self.likelihood = likelihood

        if prior is None:
            self.prior = [1.0 / NUM_STRENGTH_BUCKETS] * NUM_STRENGTH_BUCKETS
        else:
            if len(prior) != NUM_STRENGTH_BUCKETS:
                raise ValueError("Prior must have length 3.")
            s = sum(prior)
            if s <= 0:
                raise ValueError("Prior probabilities must sum to positive value.")
            self.prior = [p / s for p in prior]

        self.posterior = list(self.prior)

    def reset(self) -> None:
        """Reset posterior to prior at the start of a new hand."""
        self.posterior = list(self.prior)

    def update(self, action: int, bet_level: int) -> List[float]:
        """
        Update posterior after observing opponent action.

        Args:
            action: observed opponent action
            bet_level: current betting level

        Returns:
            posterior list [P(weak), P(mid), P(strong)]
        """
        likelihoods = [
            self.likelihood.prob(action, h, bet_level)
            for h in range(NUM_STRENGTH_BUCKETS)
        ]

        unnormalized = [
            likelihoods[h] * self.posterior[h]
            for h in range(NUM_STRENGTH_BUCKETS)
        ]

        denom = sum(unnormalized)

        if denom <= 1e-12:
            # Fallback to prior if numerical issue occurs.
            self.posterior = list(self.prior)
        else:
            self.posterior = [x / denom for x in unnormalized]

        return list(self.posterior)

    def get_probs(self) -> List[float]:
        """Return posterior probabilities [weak, mid, strong]."""
        return list(self.posterior)

    def get_label(self) -> int:
        """Return most likely strength bucket."""
        best_idx = max(
            range(NUM_STRENGTH_BUCKETS),
            key=lambda i: self.posterior[i],
        )
        return int(best_idx)

    def get_label_name(self) -> str:
        """Return most likely strength bucket name."""
        return strength_bucket_name(self.get_label())

    def __repr__(self) -> str:
        probs = self.posterior
        return (
            f"BayesianOpponentModel("
            f"weak={probs[0]:.3f}, mid={probs[1]:.3f}, strong={probs[2]:.3f}, "
            f"label={self.get_label_name()})"
        )
