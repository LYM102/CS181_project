# game/engine.py - Game engine: dealing, betting rounds, showdown full flow

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
    """Single player's state within a hand"""
    chips: int = STARTING_CHIPS
    hole_cards: list[int] = field(default_factory=list)
    round_bet: int = 0        # Cumulative bet in current betting round
    total_bet: int = 0        # Cumulative bet in this hand
    folded: bool = False
    acted_this_round: bool = False


@dataclass
class Observation:
    """
    Observable game state information for Agent.

    Following MDP modeling: s = (H_code, P_code, B_level, Pos)
    """
    hole_cards: list[int]          # Own hole cards
    community_cards: list[int]     # Community cards
    pot: int                       # Total pot
    current_bet: int               # Current amount to call
    player_round_bet: int          # Own bet in current round
    player_chips: int              # Own remaining chips
    opponent_chips: int            # Opponent remaining chips
    betting_level: int             # Betting level B_level ∈ {0,1,2,3}
    current_round: int             # Current round (PREFLOP/FLOP/TURN/RIVER)
    position: int                  # Seat position 0 or 1
    dealer_pos: int                # Dealer position
    legal_actions: list[int]       # Legal action list
    raises_this_round: int         # Number of raises in this round
    # Current hand equity (optional, computed on demand)
    equity: float = 0.0

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
    """Result of a single hand"""
    winner: Optional[int]    # Winner (0 or 1), None for tie
    pot: int                 # Pot amount
    hand_class: str          # Winner's hand class
    player_hands: dict       # {player_id: (rank, class_str)}
    rewards: dict            # {player_id: reward}


class GameEngine:
    """
    Minimalist heads-up Texas Hold'em game engine.

    Supports:
      - step-based interface: reset_hand() + step(action) → for RL training
      - hand-based interface: run_hand() → for evaluation and tournaments
      - batch run: run(num_hands) → for statistical comparison
    """

    def __init__(self, agent0, agent1):
        """
        Args:
            agent0: Agent for player 0
            agent1: Agent for player 1
        """
        self.agents = [agent0, agent1]
        self.deck = Deck()
        self.hand_count = 0

        # Game state
        self.players: list[PlayerState] = []
        self.community_cards: list[int] = []
        self.pot: int = 0
        self.current_round: int = PREFLOP
        self.betting_level: int = 0       # Current betting level
        self.raises_this_round: int = 0   # Raises in this round
        self.dealer_pos: int = 0
        self.current_player: int = 0
        self.hand_over: bool = False

    # ==================== Core Game Flow ====================

    def reset_hand(self) -> Observation:
        """
        Reset and start a new hand: shuffle, deal hole cards, post blinds.

        Returns:
            Observation for the first player to act
        """
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.hand_over = False
        self.current_round = PREFLOP

        # Save chips from previous hand
        prev_chips = [p.chips for p in self.players] if self.players else [
            STARTING_CHIPS] * NUM_PLAYERS

        # Initialize player states
        self.players = []
        for i in range(NUM_PLAYERS):
            self.players.append(PlayerState(chips=prev_chips[i]))

        # Alternate dealer position
        self.dealer_pos = self.hand_count % NUM_PLAYERS
        self.hand_count += 1

        sb_player = self.dealer_pos          # heads-up: dealer = small blind
        bb_player = 1 - self.dealer_pos      # non-dealer = big blind

        # Post blinds
        sb_amount = min(SMALL_BLIND, self.players[sb_player].chips)
        bb_amount = min(BIG_BLIND, self.players[bb_player].chips)

        self.players[sb_player].chips -= sb_amount
        self.players[bb_player].chips -= bb_amount
        self.players[sb_player].round_bet = sb_amount
        self.players[bb_player].round_bet = bb_amount
        self.players[sb_player].total_bet = sb_amount
        self.players[bb_player].total_bet = bb_amount
        self.pot = sb_amount + bb_amount

        # Deal hole cards
        for i in range(NUM_PLAYERS):
            self.players[i].hole_cards = self.deck.deal(HOLE_CARDS_PER_PLAYER)

        # Preflop: in heads-up, small blind acts first
        self.betting_level = 0
        self.raises_this_round = 0
        self.current_player = sb_player

        # Mark big blind as having a bet but not yet acted
        self.players[sb_player].acted_this_round = False
        self.players[bb_player].acted_this_round = False

        return self._get_observation(self.current_player)

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        """
        Execute current player's action.

        Args:
            action: FOLD / CALL / RAISE

        Returns:
            (observation, reward, done, info)
            - observation: Observation for the next player to act (or final state)
            - reward: Immediate reward for current player (sparse, usually 0)
            - done: Whether this hand is over
            - info: Additional information dict
        """
        if self.hand_over:
            raise RuntimeError(
                "Hand is already over. Call reset_hand() first.")

        player = self.current_player
        other = 1 - player

        if action not in self.get_legal_actions():
            raise ValueError(
                f"Player {player}: illegal action {ACTION_NAMES.get(action, action)}. "
                f"Legal: {[ACTION_NAMES[a] for a in self.get_legal_actions()]}"
            )

        info = {"action": action, "player": player,
                "action_name": ACTION_NAMES[action]}

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
            # First bet (post-flop with no existing bet) does not count as raise
            if self.betting_level < 0:
                # Opening bet: set to level 0, does not count as raise
                self.betting_level = 0
            else:
                self.raises_this_round += 1
                self.betting_level += 1

            new_bet = BETTING_LEVELS[self.betting_level]
            raise_amount = new_bet - self.players[player].round_bet
            raise_amount = max(
                0, min(raise_amount, self.players[player].chips))
            self._player_bet(player, raise_amount)
            self.players[player].acted_this_round = True
            # After raise, opponent needs to act again
            self.players[other].acted_this_round = False

        # ---- Check if betting round is over ----
        if self._is_round_over():
            if self.current_round == RIVER:
                # Last round over, proceed to showdown
                self.hand_over = True
                result = self._resolve_hand()
                obs = self._get_observation(player)
                reward = result.rewards.get(player, 0)
                return obs, reward, True, {**info, "result": result}
            else:
                # Advance to next round
                self._advance_round()
                obs = self._get_observation(self.current_player)
                return obs, 0.0, False, info
        else:
            # Switch to opponent's turn
            self.current_player = other
            obs = self._get_observation(self.current_player)
            return obs, 0.0, False, info

    def get_legal_actions(self) -> list[int]:
        """Return legal actions for the current player"""
        actions = [FOLD, CALL]
        # Check if raising is still allowed
        if self.raises_this_round < MAX_RAISES:
            # Also check if player has enough chips to raise
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
        Run a complete hand until showdown or one player folds.

        Returns:
            HandResult: result of this hand
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
        Run multiple hands in batch.

        Args:
            num_hands: number of hands to play

        Returns:
            List of results for each hand
        """
        results = []
        for _ in range(num_hands):
            result = self.run_hand()
            results.append(result)
            # Retain chips to next hand (reset if player goes bankrupt)
            for i in range(NUM_PLAYERS):
                if self.players[i].chips <= 0:
                    self.players[i].chips = STARTING_CHIPS
        return results

    # ==================== Internal Methods ====================

    def _player_bet(self, player: int, amount: int) -> None:
        """Player places a bet (moves chips to pot)"""
        actual = min(amount, self.players[player].chips)
        self.players[player].chips -= actual
        self.players[player].round_bet += actual
        self.players[player].total_bet += actual
        self.pot += actual

    def _is_round_over(self) -> bool:
        """
        Check if current betting round is over.

        Conditions:
        1. If any player is all-in (chips == 0), the round ends immediately.
        2. Both players have acted in this round AND their bets are equal.
        """
        # All‑in shortcut
        if self.players[0].chips == 0 or self.players[1].chips == 0:
            return True

        all_acted = all(p.acted_this_round for p in self.players)
        bets_equal = self.players[0].round_bet == self.players[1].round_bet
        return all_acted and bets_equal

    def _advance_round(self) -> None:
        """Advance from current round to next (deal community cards + reset betting state)"""
        self.current_round += 1

        # Reset per-round betting state
        for p in self.players:
            p.round_bet = 0
            p.acted_this_round = False
        self.raises_this_round = 0

        # Deal community cards
        if self.current_round == FLOP:
            self.community_cards += self.deck.deal(3)
            self.betting_level = -1  # No one has bet yet
        elif self.current_round in (TURN, RIVER):
            self.community_cards += self.deck.deal(1)
            self.betting_level = -1

        # Post-flop: non-dealer acts first (big blind in heads-up)
        self.current_player = 1 - self.dealer_pos

    def _resolve_hand(self) -> HandResult:
        """
        Showdown settlement: evaluate hands, distribute pot.

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
                player_hands[i] = (
                    None, "Folded" if self.players[i].folded else "N/A")

        # Determine winner
        active = [i for i in range(NUM_PLAYERS) if not self.players[i].folded]

        if len(active) == 1:
            # One player folded
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
                winner = None  # Tie
        else:
            winner = None
            hand_class = "N/A"

        # Distribute pot
        if winner is not None:
            self.players[winner].chips += self.pot
            for i in range(NUM_PLAYERS):
                if i == winner:
                    rewards[i] = self.pot
                else:
                    rewards[i] = -self.players[i].total_bet
        else:
            # Tie: split pot
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
        Build observation for the given player.

        Corresponds to MDP: s = (H_code, P_code, B_level, Pos)
        """
        other = 1 - player

        # Calculate current amount to call
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

        # Compute equity on demand (when community cards are available)
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
        """Return legal actions for the specified player"""
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

    # ==================== Display & Debug ====================

    def display_state(self) -> str:
        """Return a human-readable string of the current game state"""
        lines = []
        lines.append(
            f"=== Round: {ROUND_NAMES.get(self.current_round, '?')} ===")
        lines.append(
            f"Pot: {self.pot}  |  Betting Level: {self.betting_level}  |  Raises: {self.raises_this_round}")
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
