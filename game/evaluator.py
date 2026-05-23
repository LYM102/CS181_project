# game/evaluator.py - Hand evaluation and comparison (based on treys library)

from itertools import combinations
from treys import Card, Evaluator
from game.card import build_full_deck


# Global evaluator (treys Evaluator is stateless, can be reused)
_evaluator = Evaluator()


def evaluate_hand(hole_cards: list[int], community_cards: list[int]) -> tuple[int, str]:
    """
    Evaluate hand strength.

    Args:
        hole_cards: player's 2 hole cards (treys integer list)
        community_cards: community card list (treys integer list)

    Returns:
        (rank, hand_class_str): lower rank means stronger hand, hand_class_str is hand type name
    """
    if len(community_cards) < 3:
        raise ValueError(
            f"Need at least 3 community cards to evaluate, got {len(community_cards)}")
    rank = _evaluator.evaluate(community_cards, hole_cards)
    hand_class = _evaluator.get_rank_class(rank)
    class_str = _evaluator.class_to_string(hand_class)
    return rank, class_str


def compare_hands(
    hole_cards_1: list[int],
    hole_cards_2: list[int],
    community_cards: list[int],
) -> tuple[int, str]:
    """
    Compare two hands.

    Args:
        hole_cards_1: player 1's hole cards
        hole_cards_2: player 2's hole cards
        community_cards: community cards

    Returns:
        (result, winning_hand_class):
            result = 1  means player 1 wins
            result = -1 means player 2 wins
            result = 0  means tie
            winning_hand_class is the winner's hand type name
    """
    rank1, class1 = evaluate_hand(hole_cards_1, community_cards)
    rank2, class2 = evaluate_hand(hole_cards_2, community_cards)

    if rank1 < rank2:   # treys: lower rank = stronger
        return 1, class1
    elif rank1 > rank2:
        return -1, class2
    else:
        return 0, "Tie"


def compute_equity(
    hole_cards: list[int],
    community_cards: list[int],
    num_simulations: int = 0,
) -> float:
    """
    Compute hand win rate (equity).

    For the 16-card small deck, performs exact enumeration when num_simulations=0;
    otherwise uses Monte Carlo sampling.

    Args:
        hole_cards: own hole cards
        community_cards: revealed community cards
        num_simulations: Monte Carlo sample count, 0 means exact enumeration

    Returns:
        equity: win rate [0, 1]
    """
    known_cards = set(hole_cards + community_cards)
    remaining = [c for c in build_full_deck() if c not in known_cards]

    cards_to_deal = 5 - len(community_cards)

    if num_simulations == 0:
        return _compute_equity_exact(hole_cards, community_cards, remaining, cards_to_deal)
    else:
        return _compute_equity_mc(hole_cards, community_cards, remaining, cards_to_deal, num_simulations)


def _compute_equity_exact(
    hole_cards: list[int],
    community_cards: list[int],
    remaining: list[int],
    cards_to_deal: int,
) -> float:
    """Exact enumeration to compute equity"""
    wins = 0
    ties = 0
    total = 0

    # Enumerate all possible opponent hands + remaining community card completions
    for opponent_hand in combinations(remaining, 2):
        opp_cards = list(opponent_hand)
        remaining_after_opp = [c for c in remaining if c not in set(opp_cards)]

        if cards_to_deal == 0:
            # Community cards complete, compare directly
            full_community = community_cards
        else:
            for extra in combinations(remaining_after_opp, cards_to_deal):
                full_community = community_cards + list(extra)
                result, _ = compare_hands(
                    hole_cards, opp_cards, full_community)
                if result == 1:
                    wins += 1
                elif result == 0:
                    ties += 1
                total += 1
            continue

        result, _ = compare_hands(hole_cards, opp_cards, full_community)
        if result == 1:
            wins += 1
        elif result == 0:
            ties += 1
        total += 1

    return (wins + ties * 0.5) / total if total > 0 else 0.0


def _compute_equity_mc(
    hole_cards: list[int],
    community_cards: list[int],
    remaining: list[int],
    cards_to_deal: int,
    num_simulations: int,
) -> float:
    """Monte Carlo sampling to compute equity"""
    import random

    wins = 0
    ties = 0

    for _ in range(num_simulations):
        sampled = random.sample(remaining, 2 + cards_to_deal)
        opp_cards = sampled[:2]
        extra_community = sampled[2:2 + cards_to_deal]
        full_community = community_cards + extra_community

        result, _ = compare_hands(hole_cards, opp_cards, full_community)
        if result == 1:
            wins += 1
        elif result == 0:
            ties += 1

    return (wins + ties * 0.5) / num_simulations


def equity_to_bin(equity: float, num_bins: int = 20) -> int:
    """
    Discretize equity into bin index (for Q-table state encoding).

    Args:
        equity: win rate [0, 1]
        num_bins: number of bins

    Returns:
        bin index: 0 ~ num_bins-1
    """
    return min(int(equity * num_bins), num_bins - 1)


# Pot size thresholds for discretization
POT_BINS = [30, 60, 120, 240, 480]


def pot_to_bin(pot: int) -> int:
    """
    Discretize pot size into bin index (for Q-table state encoding).

    Bins: [0, 30], (30, 60], (60, 120], (120, 240], (240, 480], (480, +inf)
    Total 6 bins (indices 0~5).

    Args:
        pot: current total pot size

    Returns:
        bin index: 0 ~ 5
    """
    for i, threshold in enumerate(POT_BINS):
        if pot <= threshold:
            return i
    return len(POT_BINS)
