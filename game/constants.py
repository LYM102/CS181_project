# game/constants.py - Minimalist Texas Hold'em 游戏常量

# ==================== 牌组设定 ====================
SUITS = ['s', 'h']  # spades, hearts (两花色)
RANKS = ['7', '8', '9', 'T', 'J', 'Q', 'K', 'A']  # 8种点数
DECK_SIZE = 16  # 2花色 × 8点数

# ==================== 筹码与盲注 ====================
STARTING_CHIPS = 1000
SMALL_BLIND = 5
BIG_BLIND = 10

# ==================== 下注等级 ====================
BETTING_LEVELS = [10, 20, 40, 80]  # B_level 0~3 对应的下注额
MAX_RAISES = 3  # 每轮最大加注次数

# ==================== 动作编码 ====================
FOLD = 0
CALL = 1
RAISE = 2
ACTION_NAMES = {FOLD: "Fold", CALL: "Call", RAISE: "Raise"}

# ==================== 下注轮次 ====================
PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
ROUND_NAMES = {PREFLOP: "Preflop", FLOP: "Flop", TURN: "Turn", RIVER: "River"}

# ==================== 玩家与牌数 ====================
NUM_PLAYERS = 2
HOLE_CARDS_PER_PLAYER = 2
COMMUNITY_CARDS_TOTAL = 5  # 3(flop) + 1(turn) + 1(river)

# ==================== 手牌等级 (treys标准) ====================
HAND_RANK_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush"
]
