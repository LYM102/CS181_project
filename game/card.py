# game/card.py - 扑克牌与牌组 (基于 treys 库)

import random
from treys import Card
from game.constants import SUITS, RANKS, DECK_SIZE


class Deck:
    """16张极简牌组：两花色(s/h) × 八点数(7~A)"""

    def __init__(self):
        self.cards: list[int] = []
        self.reset()

    def reset(self) -> None:
        """重建并洗牌"""
        self.cards = [Card.new(f"{rank}{suit}") for suit in SUITS for rank in RANKS]
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> list[int]:
        """从牌堆顶部发 n 张牌"""
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self.cards)} remaining")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"Deck({len(self.cards)} cards remaining)"


# ==================== 卡牌工具函数 ====================

def card_to_str(card: int) -> str:
    """将 treys 卡牌整数转为可读字符串，如 '7s', 'Th', 'As'"""
    return Card.int_to_str(card)


def card_to_pretty(card: int) -> str:
    """将 treys 卡牌整数转为美化字符串，如 '7♠', 'T♥'"""
    return Card.int_to_pretty_str(card)


def cards_to_strs(cards: list[int]) -> list[str]:
    """批量转换卡牌为字符串列表"""
    return [card_to_str(c) for c in cards]


def cards_to_pretty(cards: list[int]) -> list[str]:
    """批量转换卡牌为美化字符串列表"""
    return [card_to_pretty(c) for c in cards]


def build_full_deck() -> list[int]:
    """构建完整16张牌的列表 (不发牌，仅生成)"""
    return [Card.new(f"{rank}{suit}") for suit in SUITS for rank in RANKS]
