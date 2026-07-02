"""External-sampling MCCFR solver for abstracted heads-up Texas Hold'em."""

from __future__ import annotations
import random
from collections import defaultdict
from treys import Evaluator

from game.constants import (
    BETTING_LEVELS, MAX_RAISES,
    FOLD, CALL, RAISE, SMALL_BLIND, BIG_BLIND,
    PREFLOP, FLOP, TURN, RIVER,
)
from game.card import build_full_deck
from game.evaluator import compute_hand_strength, HAND_STRENGTH_SAMPLES

_evaluator = Evaluator()
FULL_DECK = build_full_deck()


def equity_to_bucket(score: float, num_buckets: int = 10) -> int:
    """
    Map a [0,1] hand strength score to a bucket index via uniform linear mapping.
    Used for both SARSA's equity_to_bin and CFR's info set key.
    """
    return min(int(score * num_buckets), num_buckets - 1)


class CFRSolver:
    """
    External Sampling MCCFR solver for standard 52-card Texas Hold'em.

    Usage:
        solver = CFRSolver()
        solver.train(iterations=50000)
        policy = solver.get_policy()
    """

    NUM_HOLE_BUCKETS = 10

    def __init__(self):
        self.regret_table: dict[tuple, list[float]
                                ] = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.strategy_table: dict[tuple, list[float]
                                  ] = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.iteration = 0

    def _info_set_key(
        self, player: int, hole_cards: list[int], community_cards: list[int],
        betting_round: int, betting_level: int, raises_this_round: int,
    ) -> tuple:
        # Treys hand strength for ALL rounds (identical to SARSA's _encode_state).
        # - Preflop: 2 known + random 5 community → treys evaluate
        # - Postflop: known cards + random completion to 7 → treys evaluate
        hole_strength = compute_hand_strength(
            hole_cards, community_cards, num_samples=HAND_STRENGTH_SAMPLES)
        hole_bucket = equity_to_bucket(hole_strength, self.NUM_HOLE_BUCKETS)
        community_bucket = len(community_cards)
        return (player, hole_bucket, community_bucket,
                betting_round, betting_level, raises_this_round)

    def _get_strategy(self, info_key: tuple, legal_actions: list[int]) -> list[float]:
        """Regret-matching strategy"""
        regrets = self.regret_table[info_key]
        strategy = [0.0, 0.0, 0.0]
        pos_regrets = [max(0.0, regrets[a]) for a in legal_actions]
        total = sum(pos_regrets)
        if total > 0:
            for i, a in enumerate(legal_actions):
                strategy[a] = pos_regrets[i] / total
        else:
            uniform = 1.0 / len(legal_actions)
            for a in legal_actions:
                strategy[a] = uniform
        return strategy

    @staticmethod
    def _get_legal_actions(betting_level: int, raises_this_round: int) -> list[int]:
        actions = [FOLD, CALL]
        if raises_this_round < MAX_RAISES:
            next_level = betting_level + 1 if betting_level >= 0 else 0
            if next_level < len(BETTING_LEVELS):
                actions.append(RAISE)
        return actions


    def _cfr(
        self,
        hole_cards: list[list[int]],
        community_cards: list[int],
        betting_round: int,
        betting_level: int,
        raises_this_round: int,
        p0_round_bet: int, p1_round_bet: int,
        p0_total_bet: int, p1_total_bet: int,
        current_player: int,
        p0_acted: bool, p1_acted: bool,
        reach_probs: list[float],
        remaining_deck: list[int],
        traversing_player: int,  # The player being traversed (reference for opponent sampling)
    ) -> list[float]:
        """External Sampling MCCFR recursion"""

        # Check if betting round is over
        if p0_acted and p1_acted and p0_round_bet == p1_round_bet:
            if betting_round == RIVER:
                return self._showdown(hole_cards, community_cards, p0_total_bet, p1_total_bet)
            else:
                return self._advance_round(
                    hole_cards, community_cards, betting_round,
                    p0_total_bet, p1_total_bet, reach_probs,
                    remaining_deck, traversing_player,
                )

        legal_actions = self._get_legal_actions(
            betting_level, raises_this_round)
        info_key = self._info_set_key(
            current_player, hole_cards[current_player], community_cards,
            betting_round, betting_level, raises_this_round,
        )
        strategy = self._get_strategy(info_key, legal_actions)

        if current_player == traversing_player:
            # Accumulate average strategy
            for a in legal_actions:
                self.strategy_table[info_key][a] += reach_probs[current_player] * strategy[a]

            util = [0.0, 0.0]
            cf_reach = reach_probs[1 - current_player]

            for a in legal_actions:
                new_reach = list(reach_probs)
                new_reach[current_player] *= strategy[a]

                child_util = self._apply_action(
                    a, hole_cards, community_cards, betting_round, betting_level,
                    raises_this_round, p0_round_bet, p1_round_bet,
                    p0_total_bet, p1_total_bet, current_player,
                    p0_acted, p1_acted, new_reach, remaining_deck,
                    traversing_player,
                )

                for p in range(2):
                    util[p] += strategy[a] * child_util[p]

                regret = child_util[current_player] - util[current_player]
                self.regret_table[info_key][a] += cf_reach * regret

            return util

        else:
            # Sample according to strategy
            r = random.random()
            cum = 0.0
            sampled_action = legal_actions[-1]
            for a in legal_actions:
                cum += strategy[a]
                if r <= cum:
                    sampled_action = a
                    break

            # Accumulate average strategy (opponent also needs recording)
            for a in legal_actions:
                self.strategy_table[info_key][a] += reach_probs[current_player] * strategy[a]

            new_reach = list(reach_probs)
            new_reach[current_player] *= strategy[sampled_action]

            return self._apply_action(
                sampled_action, hole_cards, community_cards, betting_round,
                betting_level, raises_this_round,
                p0_round_bet, p1_round_bet, p0_total_bet, p1_total_bet,
                current_player, p0_acted, p1_acted,
                new_reach, remaining_deck, traversing_player,
            )

    def _apply_action(
        self, action: int,
        hole_cards: list[list[int]], community_cards: list[int],
        betting_round: int, betting_level: int, raises_this_round: int,
        p0_round_bet: int, p1_round_bet: int,
        p0_total_bet: int, p1_total_bet: int,
        current_player: int, p0_acted: bool, p1_acted: bool,
        reach_probs: list[float], remaining_deck: list[int],
        traversing_player: int,
    ) -> list[float]:
        """Apply action and recurse"""
        next_player = 1 - current_player

        if action == FOLD:
            pot = p0_total_bet + p1_total_bet
            if current_player == 0:
                return [float(-p0_total_bet), float(pot - p1_total_bet)]
            else:
                return [float(pot - p0_total_bet), float(-p1_total_bet)]

        elif action == CALL:
            current_bet = BETTING_LEVELS[betting_level] if betting_level >= 0 else 0
            if current_player == 0:
                call_amt = max(0, current_bet - p0_round_bet)
                return self._cfr(
                    hole_cards, community_cards, betting_round, betting_level,
                    raises_this_round,
                    p0_round_bet + call_amt, p1_round_bet,
                    p0_total_bet + call_amt, p1_total_bet,
                    next_player, True, p1_acted,
                    reach_probs, remaining_deck, traversing_player,
                )
            else:
                call_amt = max(0, current_bet - p1_round_bet)
                return self._cfr(
                    hole_cards, community_cards, betting_round, betting_level,
                    raises_this_round,
                    p0_round_bet, p1_round_bet + call_amt,
                    p0_total_bet, p1_total_bet + call_amt,
                    next_player, p0_acted, True,
                    reach_probs, remaining_deck, traversing_player,
                )

        elif action == RAISE:
            if betting_level < 0:
                new_level = 0
                new_raises = 0
            else:
                new_level = betting_level + 1
                new_raises = raises_this_round + 1
            new_bet = BETTING_LEVELS[new_level]

            if current_player == 0:
                raise_amt = max(0, new_bet - p0_round_bet)
                return self._cfr(
                    hole_cards, community_cards, betting_round, new_level,
                    new_raises,
                    p0_round_bet + raise_amt, p1_round_bet,
                    p0_total_bet + raise_amt, p1_total_bet,
                    next_player, True, False,
                    reach_probs, remaining_deck, traversing_player,
                )
            else:
                raise_amt = max(0, new_bet - p1_round_bet)
                return self._cfr(
                    hole_cards, community_cards, betting_round, new_level,
                    new_raises,
                    p0_round_bet, p1_round_bet + raise_amt,
                    p0_total_bet, p1_total_bet + raise_amt,
                    next_player, False, True,
                    reach_probs, remaining_deck, traversing_player,
                )

        return [0.0, 0.0]

    def _advance_round(
        self, hole_cards: list[list[int]], community_cards: list[int],
        betting_round: int, p0_total_bet: int, p1_total_bet: int,
        reach_probs: list[float], remaining_deck: list[int],
        traversing_player: int,
    ) -> list[float]:
        """Advance to next round (chance sampling: sample community cards)"""
        new_round = betting_round + 1
        num_new = {FLOP: 3, TURN: 1, RIVER: 1}.get(new_round, 0)
        if num_new == 0:
            return [0.0, 0.0]

        new_community = random.sample(remaining_deck, num_new)
        new_remaining = [
            c for c in remaining_deck if c not in set(new_community)]
        new_community_cards = community_cards + new_community

        first_player = 0 if self.iteration % 2 == 0 else 1

        return self._cfr(
            hole_cards, new_community_cards, new_round,
            -1, 0, 0, 0,
            p0_total_bet, p1_total_bet,
            first_player, False, False,
            reach_probs, new_remaining, traversing_player,
        )

    def _showdown(
        self, hole_cards: list[list[int]], community_cards: list[int],
        p0_total_bet: int, p1_total_bet: int,
    ) -> list[float]:
        pot = p0_total_bet + p1_total_bet
        rank0 = _evaluator.evaluate(community_cards, hole_cards[0])
        rank1 = _evaluator.evaluate(community_cards, hole_cards[1])
        if rank0 < rank1:
            return [float(pot - p0_total_bet), float(-p1_total_bet)]
        elif rank0 > rank1:
            return [float(-p0_total_bet), float(pot - p1_total_bet)]
        else:
            half = pot / 2.0
            return [half - p0_total_bet, half - p1_total_bet]


    def train(self, iterations: int = 50000, log_interval: int = 5000) -> dict:
        """Run CFR training (External Sampling MCCFR)"""
        stats = {"iterations": 0, "info_sets": 0}

        for i in range(iterations):
            self.iteration = i

            deck = list(FULL_DECK)
            random.shuffle(deck)
            hole0 = [deck[0], deck[1]]
            hole1 = [deck[2], deck[3]]
            remaining = deck[4:]

            # Alternate traversing player
            traversing_player = i % 2

            sb_player = 0 if i % 2 == 0 else 1
            p0_round_bet = SMALL_BLIND if sb_player == 0 else BIG_BLIND
            p1_round_bet = BIG_BLIND if sb_player == 0 else SMALL_BLIND

            self._cfr(
                hole_cards=[hole0, hole1],
                community_cards=[],
                betting_round=PREFLOP,
                betting_level=0,
                raises_this_round=0,
                p0_round_bet=p0_round_bet,
                p1_round_bet=p1_round_bet,
                p0_total_bet=p0_round_bet,
                p1_total_bet=p1_round_bet,
                current_player=sb_player,
                p0_acted=False,
                p1_acted=False,
                reach_probs=[1.0, 1.0],
                remaining_deck=remaining,
                traversing_player=traversing_player,
            )

            if (i + 1) % log_interval == 0:
                n_info = len(self.regret_table)
                print(f"  CFR Iter {i+1}/{iterations} | Info sets: {n_info}")

        stats["iterations"] = iterations
        stats["info_sets"] = len(self.regret_table)
        return stats


    def get_policy(self) -> dict[tuple, list[float]]:
        """Get the trained average strategy"""
        policy = {}
        for info_key in self.strategy_table:
            sums = self.strategy_table[info_key]
            legal = [a for a in range(3) if sums[a] > 0]
            if not legal:
                legal = [FOLD, CALL, RAISE]
            total = sum(sums[a] for a in legal)
            probs = [0.0, 0.0, 0.0]
            if total > 0:
                for a in legal:
                    probs[a] = sums[a] / total
            else:
                uniform = 1.0 / len(legal)
                for a in legal:
                    probs[a] = uniform
            policy[info_key] = probs
        return policy

    def get_action_prob(
        self, player: int, hole_cards: list[int], community_cards: list[int],
        betting_round: int, betting_level: int, raises_this_round: int,
        legal_actions: list[int],
    ) -> list[float]:
        """Query strategy probabilities for a specific information set"""
        info_key = self._info_set_key(
            player, hole_cards, community_cards,
            betting_round, betting_level, raises_this_round,
        )

        if info_key in self.strategy_table:
            sums = self.strategy_table[info_key]
            total = sum(sums[a] for a in legal_actions)
            if total > 0:
                probs = [0.0, 0.0, 0.0]
                for a in legal_actions:
                    probs[a] = sums[a] / total
                return probs

        # Heuristic fallback (uses treys hand strength, thresholds adjusted
        # for treys scale where strong ~0.67, medium ~0.52, weak ~0.41)
        hs = compute_hand_strength(
            hole_cards, community_cards, num_samples=HAND_STRENGTH_SAMPLES)
        probs = [0.0, 0.0, 0.0]
        if hs > 0.60 and RAISE in legal_actions:
            probs[RAISE] = 0.6
            probs[CALL] = 0.35
            probs[FOLD] = 0.05
        elif hs > 0.48:
            probs[CALL] = 0.7
            probs[FOLD] = 0.2
            probs[RAISE] = 0.1
        else:
            probs[FOLD] = 0.5
            probs[CALL] = 0.4
            probs[RAISE] = 0.1
        total = sum(probs[a] for a in legal_actions)
        if total > 0:
            for a in legal_actions:
                probs[a] /= total
            for a in range(3):
                if a not in legal_actions:
                    probs[a] = 0.0
        return probs

    def save_policy(self, filepath: str) -> None:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'regret_table': dict(self.regret_table),
                'strategy_table': dict(self.strategy_table),
            }, f)
        print(
            f"Policy saved to {filepath} ({len(self.strategy_table)} info sets)")

    def load_policy(self, filepath: str) -> None:
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.regret_table = defaultdict(
            lambda: [0.0, 0.0, 0.0], data['regret_table'])
        self.strategy_table = defaultdict(
            lambda: [0.0, 0.0, 0.0], data['strategy_table'])
        print(
            f"Policy loaded from {filepath} ({len(self.strategy_table)} info sets)")
