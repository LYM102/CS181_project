# game/evaluator.py - 手牌评估与比较 (基于 treys 库)

from itertools import combinations
from treys import Card, Evaluator
from game.card import build_full_deck


# 全局评估器 (treys Evaluator 是无状态的，可复用)
_evaluator = Evaluator()


def evaluate_hand(hole_cards: list[int], community_cards: list[int]) -> tuple[int, str]:
    """
    评估手牌强度。

    Args:
        hole_cards: 玩家的2张底牌 (treys整数列表)
        community_cards: 公共牌列表 (treys整数列表)

    Returns:
        (rank, hand_class_str): rank越小牌力越强, hand_class_str为牌型名称
    """
    if len(community_cards) < 3:
        raise ValueError(f"Need at least 3 community cards to evaluate, got {len(community_cards)}")
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
    比较两手牌的大小。

    Args:
        hole_cards_1: 玩家1的底牌
        hole_cards_2: 玩家2的底牌
        community_cards: 公共牌

    Returns:
        (result, winning_hand_class):
            result = 1  表示玩家1赢
            result = -1 表示玩家2赢
            result = 0  表示平局
            winning_hand_class 为赢家的牌型名称
    """
    rank1, class1 = evaluate_hand(hole_cards_1, community_cards)
    rank2, class2 = evaluate_hand(hole_cards_2, community_cards)

    if rank1 < rank2:   # treys: rank越小越强
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
    计算手牌胜率 (equity)。

    对于16张牌的小牌组，当 num_simulations=0 时进行精确枚举;
    否则进行蒙特卡洛采样。

    Args:
        hole_cards: 己方底牌
        community_cards: 已翻开的公共牌
        num_simulations: 蒙特卡洛采样次数，0表示精确枚举

    Returns:
        equity: 胜率 [0, 1]
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
    """精确枚举计算 equity"""
    wins = 0
    ties = 0
    total = 0

    # 枚举对手所有可能底牌 + 剩余公共牌补全
    for opponent_hand in combinations(remaining, 2):
        opp_cards = list(opponent_hand)
        remaining_after_opp = [c for c in remaining if c not in set(opp_cards)]

        if cards_to_deal == 0:
            # 公共牌已全，直接比较
            full_community = community_cards
        else:
            for extra in combinations(remaining_after_opp, cards_to_deal):
                full_community = community_cards + list(extra)
                result, _ = compare_hands(hole_cards, opp_cards, full_community)
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
    """蒙特卡洛采样计算 equity"""
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
    将 equity 离散化为 bin 编号 (用于 Q-table 状态编码)。

    Args:
        equity: [0, 1] 的胜率
        num_bins: 分桶数量

    Returns:
        bin index: 0 ~ num_bins-1
    """
    return min(int(equity * num_bins), num_bins - 1)
