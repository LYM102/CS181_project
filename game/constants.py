# game/constants.py - Standard Texas Hold'em game constants (52 cards)

# ==================== Deck Setup ====================
SUITS = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs (4 suits)
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']  # 13 ranks
DECK_SIZE = 52  # 4 suits × 13 ranks

# ==================== Chips & Blinds ====================
STARTING_CHIPS = 1000
SMALL_BLIND = 5
BIG_BLIND = 10

# ==================== Betting Levels ====================
BETTING_LEVELS = [10, 20, 40, 80, 160, 320]  # B_level 0~5 corresponding bet amounts
MAX_RAISES = 4  # Max raises per round

# ==================== Action Encoding ====================
FOLD = 0
CALL = 1
RAISE = 2
ACTION_NAMES = {FOLD: "Fold", CALL: "Call", RAISE: "Raise"}

# ==================== Betting Rounds ==
PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
ROUND_NAMES = {PREFLOP: "Preflop", FLOP: "Flop", TURN: "Turn", RIVER: "River"}

# ==================== Players & Card Counts ==
NUM_PLAYERS = 2
HOLE_CARDS_PER_PLAYER = 2
COMMUNITY_CARDS_TOTAL = 5  # 3(flop) + 1(turn) + 1(river)

# ==================== Hand Rankings (treys standard) ====================
HAND_RANK_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush"
]
