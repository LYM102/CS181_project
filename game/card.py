"""Deck and card utilities (treys-based, 52-card)."""

from __future__ import annotations
import random
from treys import Card
from game.constants import SUITS, RANKS, DECK_SIZE


class Deck:
    """Standard 52-card deck."""

    def __init__(self):
        self.cards: list[int] = []
        self.reset()

    def reset(self) -> None:
        """Rebuild and shuffle the deck"""
        self.cards = [Card.new(f"{rank}{suit}") for suit in SUITS for rank in RANKS]
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> list[int]:
        """Deal n cards from the top of the deck"""
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self.cards)} remaining")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"Deck({len(self.cards)} cards remaining)"


def card_to_str(card: int) -> str:
    """Convert a treys card integer to a readable string, e.g. '7s', 'Th', 'As'"""
    return Card.int_to_str(card)


def card_to_pretty(card: int) -> str:
    """Convert a treys card integer to a pretty string, e.g. '7s', 'Th'"""
    return Card.int_to_pretty_str(card)


def cards_to_strs(cards: list[int]) -> list[str]:
    """Batch convert cards to string list"""
    return [card_to_str(c) for c in cards]


def cards_to_pretty(cards: list[int]) -> list[str]:
    """Batch convert cards to pretty string list"""
    return [card_to_pretty(c) for c in cards]


def build_full_deck() -> list[int]:
    """Build the complete 52-card list (generation only, no dealing)"""
    return [Card.new(f"{rank}{suit}") for suit in SUITS for rank in RANKS]
