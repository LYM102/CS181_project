# game/evaluator.py - Hand evaluation and comparison (based on treys library, 52 cards)

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


# ==================== Basic Hand Evaluation ====================

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


# ==================== Equity Computation ====================

@lru_cache(maxsize=20000)
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


# ==================== Discretization ====================

def equity_to_bin(equity: float, bins: int = 20) -> int:
    """
    Discretize equity [0, 1] into bin index [0, bins-1].

    Uses logarithmic spacing: more granular near extremes.
    """
    if equity <= 0.0:
        return 0
    if equity >= 1.0:
        return bins - 1
    # Logarithmic mapping
    import math
    idx = int(round(math.log(equity / (1 - equity + 1e-10) + 1) / math.log(2) * 3))
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
