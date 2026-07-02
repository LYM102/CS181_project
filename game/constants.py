"""Game constants for 52-card heads-up Texas Hold'em."""

SUITS = ['s', 'h', 'd', 'c']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK_SIZE = 52

STARTING_CHIPS = 1000
SMALL_BLIND = 5
BIG_BLIND = 10

BETTING_LEVELS = [10, 20, 40, 80, 160, 320]
MAX_RAISES = 4

FOLD = 0
CALL = 1
RAISE = 2
ACTION_NAMES = {FOLD: "Fold", CALL: "Call", RAISE: "Raise"}

PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
ROUND_NAMES = {PREFLOP: "Preflop", FLOP: "Flop", TURN: "Turn", RIVER: "River"}

NUM_PLAYERS = 2
HOLE_CARDS_PER_PLAYER = 2
COMMUNITY_CARDS_TOTAL = 5

HAND_RANK_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush"
]
