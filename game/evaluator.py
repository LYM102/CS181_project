"""Hand strength evaluation and abstraction (treys-based)."""

from __future__ import annotations
from functools import lru_cache
import random
from itertools import combinations
from treys import Card, Evaluator
from game.card import build_full_deck


# Global evaluator (treys Evaluator is stateless, can be reused)
_evaluator = Evaluator()

# Full deck of 52 cards (4 suits, 2..A) for equity calculation
FULL_DECK = build_full_deck()


def evaluate_hand(hole_cards: list[int], community_cards: list[int]) -> tuple[int, str]:
    """
    Evaluate hand strength.

    Args:
        hole_cards: list of 2 treys integers (own hand)
        community_cards: list of 3/4/5 treys integers (board)

    Returns:
        (rank, class_str): lower rank = stronger hand
    """
    rank = _evaluator.evaluate(hole_cards, community_cards)
    class_str = _evaluator.get_rank_class(rank)
    return rank, class_str


def hand_class_name(rank: int) -> str:
    """Convert treys rank to human-readable hand class name"""
    return _evaluator.class_to_string(_evaluator.get_rank_class(rank))


def compare_hands(hole1: list[int], hole2: list[int],
                  community: list[int]) -> tuple[int, str]:
    """
    Compare two hands over the same community cards.

    Args:
        hole1: Player 0's hole cards
        hole2: Player 1's hole cards
        community: Community cards

    Returns:
        (1 if P0 wins, 0 if tie, -1 if P1 wins, best_hand_class_str)
    """
    rank1, cls1 = evaluate_hand(hole1, community)
    rank2, cls2 = evaluate_hand(hole2, community)

    if rank1 < rank2:
        return 1, cls1
    elif rank1 > rank2:
        return -1, cls2
    return 0, cls1


@lru_cache(maxsize=200000)
def _cached_equity(hole_tuple: tuple, comm_tuple: tuple, sim: int) -> float:
    """
    Cached equity computation with MC sampling.

    Args:
        hole_tuple: tuple of 2 card ints (for caching)
        comm_tuple: tuple of community card ints (for caching)
        sim: number of MC samples (0 = exact enumeration when feasible)

    Returns:
        win_rate: float in [0, 1]
    """
    hole = list(hole_tuple)
    community = list(comm_tuple)
    used = set(hole) | set(community)
    remaining = [c for c in FULL_DECK if c not in used]

    # For very small remaining decks (< 20 cards), exact enumeration is feasible
    if len(remaining) <= 20 and sim == 0:
        wins = 0
        total = 0
        # Enumerate opponent's cards
        for opp_hole in combinations(remaining, 2):
            opp_set = set(opp_hole)
            board_cards_needed = 5 - len(community)
            remaining_for_board = [c for c in remaining if c not in opp_set]
            for board in combinations(remaining_for_board, board_cards_needed):
                full_board = community + list(board)
                r1, _ = evaluate_hand(hole, full_board)
                r2, _ = evaluate_hand(list(opp_hole), full_board)
                total += 1
                if r1 < r2:
                    wins += 1
                elif r1 == r2:
                    wins += 0.5
        return wins / total if total > 0 else 0.5

    # Monte Carlo sampling
    wins = 0
    for _ in range(sim):
        # Sample opponent cards
        opp_sample = random.sample(remaining, 2)
        opp_set = set(opp_sample)
        remaining_for_board = [c for c in remaining if c not in opp_set]
        # Sample board cards needed
        board_cards_needed = 5 - len(community)
        if board_cards_needed > 0:
            board_sample = random.sample(remaining_for_board, board_cards_needed)
        else:
            board_sample = []
        full_board = community + board_sample

        r1, _ = evaluate_hand(hole, full_board)
        r2, _ = evaluate_hand(list(opp_sample), full_board)
        if r1 < r2:
            wins += 1
        elif r1 == r2:
            wins += 0.5

    return wins / sim if sim > 0 else 0.5


def compute_equity(hole_cards: list[int], community_cards: list[int],
                   sim: int = 100) -> float:
    """
    Compute equity (win rate) for a given hand + board.

    Args:
        hole_cards: list of 2 card ints
        community_cards: list of 0~5 community card ints
        sim: MC samples (0 for exact enumeration when feasible, default 100 for speed)

    Returns:
        equity: float in [0, 1]
    """
    return _cached_equity(tuple(sorted(hole_cards)),
                          tuple(sorted(community_cards)),
                          sim)


@lru_cache(maxsize=200000)
def _cached_hand_strength(hole_tuple: tuple, comm_tuple: tuple,
                          num_samples: int) -> float:
    """
    Cached hand strength via treys evaluator with random board completion.

    When fewer than 7 cards are known (2 hole + up to 5 community),
    randomly sample the missing community cards from the remaining deck,
    evaluate the best 5-card hand from the 7 total cards using treys,
    and average over multiple trials.

    Returns a score in [0, 1] where higher = stronger hand.
    """
    hole = list(hole_tuple)
    community = list(comm_tuple)
    known = hole + community
    used = set(known)
    remaining = [c for c in FULL_DECK if c not in used]

    cards_needed = 7 - len(known)  # 2 hole + 5 community = 7 total

    if cards_needed <= 0:
        # River: all 7 cards known, evaluate directly
        rank, _ = evaluate_hand(hole, community)
        return 1.0 - rank / 7462.0

    total_rank = 0
    for _ in range(num_samples):
        sampled = random.sample(remaining, cards_needed)
        full_community = community + sampled
        rank, _ = evaluate_hand(hole, full_community)
        total_rank += rank

    avg_rank = total_rank / num_samples
    return 1.0 - avg_rank / 7462.0


# Shared MC sample count for SARSA (GameEngine) and CFR hand-strength bucketing.
HAND_STRENGTH_SAMPLES = 100


def compute_hand_strength(hole_cards: list[int], community_cards: list[int],
                          num_samples: int = HAND_STRENGTH_SAMPLES) -> float:
    """
    Compute hand strength by completing to 7 cards and evaluating with treys.

    This is more principled than the old MC equity because:
      - It measures absolute hand strength (treys rank of best 5-card hand)
      - Random completion correctly handles unknown community cards
      - The score [0,1] is a cleaner state representation

    Examples:
      - Preflop: 2 known, sample 5 random community cards
      - Flop:    5 known, sample 2 random community cards
      - Turn:    6 known, sample 1 random community card
      - River:   7 known, no sampling needed

    Args:
        hole_cards: 2 hole card ints
        community_cards: 0~5 community card ints
        num_samples: MC samples for incomplete boards

    Returns:
        strength: float in [0, 1], higher = stronger hand
    """
    return _cached_hand_strength(tuple(sorted(hole_cards)),
                                 tuple(sorted(community_cards)),
                                 num_samples)


def equity_to_bin(equity: float, bins: int = 20) -> int:
    """
    Discretize equity [0, 1] into bin index [0, bins-1].

    Uses uniform linear mapping so that all bins are utilised.
    The previous logarithmic mapping compressed realistic preflop
    equities (0.28-0.88) into only 8 of 20 bins, wasting 55% of the
    state space and causing severe state aliasing for SARSA.
    """
    if equity <= 0.0:
        return 0
    if equity >= 1.0:
        return bins - 1
    idx = int(equity * bins)
    return max(0, min(bins - 1, idx))


def pot_to_bin(pot: int) -> int:
    """
    Discretize pot size into 7 bins (0~6).

    Bins: [0,30], (30,60], (60,120], (120,240], (240,480], (480,960], >960
    """
    thresholds = [30, 60, 120, 240, 480, 960]
    for i, t in enumerate(thresholds):
        if pot <= t:
            return i
    return len(thresholds)  # bin 6


# Calibrated for compute_hand_strength(): preflop ~[0.41, 0.67], river up to ~1.0.
HAND_STRENGTH_WEAK = 0.48
HAND_STRENGTH_STRONG = 0.62


def opponent_hand_strength(hole_cards: list[int], community_cards: list[int],
                           num_samples: int = HAND_STRENGTH_SAMPLES) -> float:
    """Opponent hand strength for BNN labels/features (same as own-hand scoring)."""
    return compute_hand_strength(hole_cards, community_cards, num_samples)


def hand_strength_to_label(strength: float) -> int:
    """Map treys hand strength to 3-class label: 0=weak, 1=mid, 2=strong."""
    if strength > HAND_STRENGTH_STRONG:
        return 2
    if strength > HAND_STRENGTH_WEAK:
        return 1
    return 0


def hand_strength_to_label_5class(strength: float) -> int:
    """Map treys hand strength to 5-class label."""
    if strength > 0.72:
        return 4   # very_strong
    if strength > 0.62:
        return 3   # strong
    if strength > 0.52:
        return 2   # mid
    if strength > 0.44:
        return 1   # weak
    return 0       # very_weak
