# game/constants.py - Minimalist Texas Hold'em game constants

# ==================== Deck Setup ====================
SUITS = ['s', 'h']  # spades, hearts (2 suits)
RANKS = ['7', '8', '9', 'T', 'J', 'Q', 'K', 'A']  # 8 ranks
DECK_SIZE = 16  # 2 suits × 8 ranks

# ==================== Chips & Blinds ====================
STARTING_CHIPS = 1000
SMALL_BLIND = 5
BIG_BLIND = 10

# ==================== Betting Levels ====================
BETTING_LEVELS = [10, 20, 40, 80]  # B_level 0~3 corresponding bet amounts
MAX_RAISES = 3  # Max raises per round

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
