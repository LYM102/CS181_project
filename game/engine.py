# game/engine.py - 游戏引擎：发牌、下注轮次、开牌全流程

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from game.card import Deck, cards_to_pretty, cards_to_strs
from game.evaluator import evaluate_hand, compare_hands, compute_equity
from game.constants import (
    STARTING_CHIPS, SMALL_BLIND, BIG_BLIND,
    BETTING_LEVELS, MAX_RAISES,
    FOLD, CALL, RAISE, ACTION_NAMES,
    PREFLOP, FLOP, TURN, RIVER, ROUND_NAMES,
    NUM_PLAYERS, HOLE_CARDS_PER_PLAYER, COMMUNITY_CARDS_TOTAL,
)


@dataclass
class PlayerState:
    """单个玩家在一手牌中的状态"""
    chips: int = STARTING_CHIPS
    hole_cards: list[int] = field(default_factory=list)
    round_bet: int = 0        # 当前下注轮的累计下注
    total_bet: int = 0        # 本手牌的累计下注
    folded: bool = False
    acted_this_round: bool = False


@dataclass
class Observation:
    """
    Agent 可观测到的游戏状态信息。

    遵循 MDP 建模: s = (H_code, P_code, B_level, Pos)
    """
    hole_cards: list[int]          # 己方底牌
    community_cards: list[int]     # 公共牌
    pot: int                       # 底池总额
    current_bet: int               # 当前需要跟注的金额
    player_round_bet: int          # 己方本轮已下注额
    player_chips: int              # 己方剩余筹码
    opponent_chips: int            # 对手剩余筹码
    betting_level: int             # 下注等级 B_level ∈ {0,1,2,3}
    current_round: int             # 当前轮次 (PREFLOP/FLOP/TURN/RIVER)
    position: int                  # 座位位置 0 or 1
    dealer_pos: int                # 庄家位置
    legal_actions: list[int]       # 合法动作列表
    raises_this_round: int         # 本轮已加注次数
    equity: float = 0.0            # 当前手牌 equity (可选，按需计算)

    @property
    def hole_cards_str(self) -> list[str]:
        return cards_to_strs(self.hole_cards)

    @property
    def community_cards_str(self) -> list[str]:
        return cards_to_strs(self.community_cards)

    @property
    def hole_cards_pretty(self) -> list[str]:
        return cards_to_pretty(self.hole_cards)

    @property
    def community_cards_pretty(self) -> list[str]:
        return cards_to_pretty(self.community_cards)


@dataclass
class HandResult:
    """一手牌的结果"""
    winner: Optional[int]    # 赢家 (0 or 1), 平局为 None
    pot: int                 # 底池金额
    hand_class: str          # 赢家牌型
    player_hands: dict       # {player_id: (rank, class_str)}
    rewards: dict            # {player_id: reward}


class GameEngine:
    """
    极简双人德州扑克游戏引擎。

    支持:
      - step-based 接口: reset_hand() + step(action) → 适合 RL 训练
      - hand-based 接口: run_hand() → 适合评估和锦标赛
      - 批量运行: run(num_hands) → 适合统计对比
    """

    def __init__(self, agent0, agent1):
        """
        Args:
            agent0: 玩家0的 Agent
            agent1: 玩家1的 Agent
        """
        self.agents = [agent0, agent1]
        self.deck = Deck()
        self.hand_count = 0

        # 游戏状态
        self.players: list[PlayerState] = []
        self.community_cards: list[int] = []
        self.pot: int = 0
        self.current_round: int = PREFLOP
        self.betting_level: int = 0       # 当前下注等级
        self.raises_this_round: int = 0   # 本轮加注次数
        self.dealer_pos: int = 0
        self.current_player: int = 0
        self.hand_over: bool = False

    # ==================== 核心游戏流程 ====================

    def reset_hand(self) -> Observation:
        """
        重置并开始新一手牌：洗牌、发底牌、下盲注。

        Returns:
            第一个行动玩家的 Observation
        """
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.hand_over = False
        self.current_round = PREFLOP

        # 保存上一手的筹码
        prev_chips = [p.chips for p in self.players] if self.players else [STARTING_CHIPS] * NUM_PLAYERS

        # 初始化玩家状态
        self.players = []
        for i in range(NUM_PLAYERS):
            self.players.append(PlayerState(chips=prev_chips[i]))

        # 交替庄家位置
        self.dealer_pos = self.hand_count % NUM_PLAYERS
        self.hand_count += 1

        sb_player = self.dealer_pos          # heads-up: 庄家=小盲
        bb_player = 1 - self.dealer_pos      # 非庄家=大盲

        # 下盲注
        sb_amount = min(SMALL_BLIND, self.players[sb_player].chips)
        bb_amount = min(BIG_BLIND, self.players[bb_player].chips)

        self.players[sb_player].chips -= sb_amount
        self.players[bb_player].chips -= bb_amount
        self.players[sb_player].round_bet = sb_amount
        self.players[bb_player].round_bet = bb_amount
        self.players[sb_player].total_bet = sb_amount
        self.players[bb_player].total_bet = bb_amount
        self.pot = sb_amount + bb_amount

        # 发底牌
        for i in range(NUM_PLAYERS):
            self.players[i].hole_cards = self.deck.deal(HOLE_CARDS_PER_PLAYER)

        # Preflop: heads-up 中小盲先行动
        self.betting_level = 0
        self.raises_this_round = 0
        self.current_player = sb_player

        # 标记大盲已"有下注"但尚未"主动行动"
        self.players[sb_player].acted_this_round = False
        self.players[bb_player].acted_this_round = False

        return self._get_observation(self.current_player)

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        """
        执行当前玩家的动作。

        Args:
            action: FOLD / CALL / RAISE

        Returns:
            (observation, reward, done, info)
            - observation: 下一个需要行动的玩家的观测 (或最终状态)
            - reward: 当前玩家的即时奖励 (稀疏奖励，通常为0)
            - done: 这手牌是否结束
            - info: 额外信息字典
        """
        if self.hand_over:
            raise RuntimeError("Hand is already over. Call reset_hand() first.")

        player = self.current_player
        other = 1 - player

        if action not in self.get_legal_actions():
            raise ValueError(
                f"Player {player}: illegal action {ACTION_NAMES.get(action, action)}. "
                f"Legal: {[ACTION_NAMES[a] for a in self.get_legal_actions()]}"
            )

        info = {"action": action, "player": player, "action_name": ACTION_NAMES[action]}

        # ---- FOLD ----
        if action == FOLD:
            self.players[player].folded = True
            self.hand_over = True
            result = self._resolve_hand()
            obs = self._get_observation(other)
            reward = result.rewards.get(player, 0)
            return obs, reward, True, {**info, "result": result}

        # ---- CALL ----
        elif action == CALL:
            current_bet = BETTING_LEVELS[self.betting_level] if self.betting_level >= 0 else 0
            call_amount = current_bet - self.players[player].round_bet
            call_amount = max(0, min(call_amount, self.players[player].chips))
            self._player_bet(player, call_amount)
            self.players[player].acted_this_round = True

        # ---- RAISE ----
        elif action == RAISE:
            # 首次下注 (post-flop 无既有下注时) 不计为 raise
            if self.betting_level < 0:
                # 开局下注: 设为 level 0, 不算加注次数
                self.betting_level = 0
            else:
                self.raises_this_round += 1
                self.betting_level += 1

            new_bet = BETTING_LEVELS[self.betting_level]
            raise_amount = new_bet - self.players[player].round_bet
            raise_amount = max(0, min(raise_amount, self.players[player].chips))
            self._player_bet(player, raise_amount)
            self.players[player].acted_this_round = True
            # 加注后对手需要重新行动
            self.players[other].acted_this_round = False

        # ---- 检查下注轮是否结束 ----
        if self._is_round_over():
            if self.current_round == RIVER:
                # 最后一轮结束，进入开牌
                self.hand_over = True
                result = self._resolve_hand()
                obs = self._get_observation(player)
                reward = result.rewards.get(player, 0)
                return obs, reward, True, {**info, "result": result}
            else:
                # 进入下一轮
                self._advance_round()
                obs = self._get_observation(self.current_player)
                return obs, 0.0, False, info
        else:
            # 切换到对手行动
            self.current_player = other
            obs = self._get_observation(self.current_player)
            return obs, 0.0, False, info

    def get_legal_actions(self) -> list[int]:
        """返回当前玩家的合法动作列表"""
        actions = [FOLD, CALL]
        # 检查是否还能加注
        if self.raises_this_round < MAX_RAISES:
            # 还需检查筹码是否够加注
            if self.betting_level < 0:
                next_level = 0
            else:
                next_level = self.betting_level + 1
            if next_level < len(BETTING_LEVELS):
                new_bet = BETTING_LEVELS[next_level]
                needed = new_bet - self.players[self.current_player].round_bet
                if needed <= self.players[self.current_player].chips:
                    actions.append(RAISE)
        return actions

    def run_hand(self) -> HandResult:
        """
        运行完整一手牌，直到开牌或一方弃牌。

        Returns:
            HandResult: 这手牌的结果
        """
        obs = self.reset_hand()
        done = False

        while not done:
            player = self.current_player
            action = self.agents[player].act(obs)
            obs, reward, done, info = self.step(action)

        return info.get("result", self._resolve_hand())

    def run(self, num_hands: int = 1000) -> list[HandResult]:
        """
        批量运行多手牌。

        Args:
            num_hands: 运行手数

        Returns:
            每手牌的结果列表
        """
        results = []
        for _ in range(num_hands):
            result = self.run_hand()
            results.append(result)
            # 保留筹码到下一手 (若玩家破产则重置)
            for i in range(NUM_PLAYERS):
                if self.players[i].chips <= 0:
                    self.players[i].chips = STARTING_CHIPS
        return results

    # ==================== 内部方法 ====================

    def _player_bet(self, player: int, amount: int) -> None:
        """玩家下注 (从筹码移至底池)"""
        actual = min(amount, self.players[player].chips)
        self.players[player].chips -= actual
        self.players[player].round_bet += actual
        self.players[player].total_bet += actual
        self.pot += actual

    def _is_round_over(self) -> bool:
        """判断当前下注轮是否结束"""
        # 如果有人弃牌，轮次结束 (但 hand_over 已经在 step 中设置了)
        # 两人都已行动 且 下注额相等
        all_acted = all(p.acted_this_round for p in self.players)
        bets_equal = self.players[0].round_bet == self.players[1].round_bet
        return all_acted and bets_equal

    def _advance_round(self) -> None:
        """从当前轮次推进到下一轮 (翻公共牌 + 重置下注状态)"""
        self.current_round += 1

        # 重置每轮下注状态
        for p in self.players:
            p.round_bet = 0
            p.acted_this_round = False
        self.raises_this_round = 0

        # 翻公共牌
        if self.current_round == FLOP:
            self.community_cards += self.deck.deal(3)
            self.betting_level = -1  # 尚未有人下注
        elif self.current_round in (TURN, RIVER):
            self.community_cards += self.deck.deal(1)
            self.betting_level = -1

        # Post-flop: 非庄家先行动 (heads-up 中是大盲位)
        self.current_player = 1 - self.dealer_pos

    def _resolve_hand(self) -> HandResult:
        """
        开牌结算: 评估手牌、分配底池。

        Returns:
            HandResult
        """
        player_hands = {}
        rewards = {}

        for i in range(NUM_PLAYERS):
            if not self.players[i].folded and len(self.community_cards) >= 3:
                rank, class_str = evaluate_hand(
                    self.players[i].hole_cards, self.community_cards
                )
                player_hands[i] = (rank, class_str)
            else:
                player_hands[i] = (None, "Folded" if self.players[i].folded else "N/A")

        # 确定赢家
        active = [i for i in range(NUM_PLAYERS) if not self.players[i].folded]

        if len(active) == 1:
            # 一方弃牌
            winner = active[0]
            hand_class = "Opponent Folded"
        elif len(active) == 2:
            result, hand_class = compare_hands(
                self.players[0].hole_cards,
                self.players[1].hole_cards,
                self.community_cards,
            )
            if result == 1:
                winner = 0
            elif result == -1:
                winner = 1
            else:
                winner = None  # 平局
        else:
            winner = None
            hand_class = "N/A"

        # 分配底池
        if winner is not None:
            self.players[winner].chips += self.pot
            for i in range(NUM_PLAYERS):
                if i == winner:
                    rewards[i] = self.pot
                else:
                    rewards[i] = -self.players[i].total_bet
        else:
            # 平局: 平分底池
            half = self.pot // 2
            for i in active:
                self.players[i].chips += half
            for i in range(NUM_PLAYERS):
                if i in active:
                    rewards[i] = half - self.players[i].total_bet
                else:
                    rewards[i] = -self.players[i].total_bet

        return HandResult(
            winner=winner,
            pot=self.pot,
            hand_class=hand_class,
            player_hands=player_hands,
            rewards=rewards,
        )

    def _get_observation(self, player: int) -> Observation:
        """
        构建给定玩家的观测信息。

        对应 MDP: s = (H_code, P_code, B_level, Pos)
        """
        other = 1 - player

        # 计算当前需要跟注的金额
        if self.betting_level >= 0:
            current_bet = BETTING_LEVELS[self.betting_level]
        else:
            current_bet = 0

        obs = Observation(
            hole_cards=list(self.players[player].hole_cards),
            community_cards=list(self.community_cards),
            pot=self.pot,
            current_bet=current_bet,
            player_round_bet=self.players[player].round_bet,
            player_chips=self.players[player].chips,
            opponent_chips=self.players[other].chips,
            betting_level=max(0, self.betting_level),
            current_round=self.current_round,
            position=player,
            dealer_pos=self.dealer_pos,
            legal_actions=self.get_legal_actions_for(player),
            raises_this_round=self.raises_this_round,
        )

        # 按需计算 equity (有公共牌时)
        if len(self.community_cards) >= 3:
            try:
                obs.equity = compute_equity(
                    self.players[player].hole_cards,
                    self.community_cards,
                )
            except Exception:
                obs.equity = 0.0

        return obs

    def get_legal_actions_for(self, player: int) -> list[int]:
        """返回指定玩家的合法动作列表"""
        actions = [FOLD, CALL]
        if self.raises_this_round < MAX_RAISES:
            if self.betting_level < 0:
                next_level = 0
            else:
                next_level = self.betting_level + 1
            if next_level < len(BETTING_LEVELS):
                new_bet = BETTING_LEVELS[next_level]
                needed = new_bet - self.players[player].round_bet
                if needed <= self.players[player].chips:
                    actions.append(RAISE)
        return actions

    # ==================== 显示与调试 ====================

    def display_state(self) -> str:
        """返回当前游戏状态的可读字符串"""
        lines = []
        lines.append(f"=== Round: {ROUND_NAMES.get(self.current_round, '?')} ===")
        lines.append(f"Pot: {self.pot}  |  Betting Level: {self.betting_level}  |  Raises: {self.raises_this_round}")
        lines.append(f"Community: {cards_to_pretty(self.community_cards)}")
        for i in range(NUM_PLAYERS):
            p = self.players[i]
            marker = " <-- Current" if i == self.current_player else ""
            dealer = " (D/SB)" if i == self.dealer_pos else " (BB)"
            folded = " [FOLDED]" if p.folded else ""
            lines.append(
                f"  Player {i}{dealer}{marker}{folded}: "
                f"Chips={p.chips}  RoundBet={p.round_bet}  "
                f"Hole={cards_to_pretty(p.hole_cards)}"
            )
        return "\n".join(lines)
