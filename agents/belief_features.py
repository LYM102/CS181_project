# agents/belief_features.py

from __future__ import annotations

from collections import defaultdict

import numpy as np

from game.engine import Observation
from game.constants import MAX_RAISES


class BeliefFeatureEncoder:
    BNN_FEATURE_DIM = 53

    def __init__(self):
        self._opp_actions: list[tuple[int, int]] = []
        self._self_actions: list[tuple[int, int]] = []
        self._prev_community_count: int = 0
        self._auto_record_self: bool = True

    def reset(self) -> None:
        self._opp_actions = []
        self._self_actions = []
        self._prev_community_count = 0

    def record_action(self, player: int, action: int, round_num: int) -> None:
        if player == getattr(self, "player_id", 0):
            self._self_actions.append((round_num, action))
        else:
            self._opp_actions.append((round_num, action))

    def _encode_bnn_features(self, obs: Observation,
                             opp_equity: float = None,
                             opp_rank_avg: float = None,
                             opp_suited: float = None) -> np.ndarray:
        features = np.zeros(self.BNN_FEATURE_DIM, dtype=np.float32)
        features[0] = obs.equity

        from treys import Card
        cc = obs.community_cards
        if len(cc) >= 3:
            ranks = [Card.get_rank_int(c) for c in cc]
            suits = [Card.get_suit_int(c) for c in cc]
            features[1] = 1.0 if len(set(ranks)) < len(ranks) else 0.0
            max_suit_count = max(suits.count(s) for s in set(suits))
            features[2] = 1.0 if max_suit_count >= 2 else 0.0
            unique_ranks = sorted(set(ranks))
            has_straight_draw = any(
                unique_ranks[i + 2] - unique_ranks[i] <= 5
                for i in range(len(unique_ranks) - 2))
            features[3] = 1.0 if has_straight_draw else 0.0
            if len(unique_ranks) >= 2:
                gaps = [unique_ranks[i + 1] - unique_ranks[i] for i in range(len(unique_ranks) - 1)]
                features[4] = 1.0 - min(sum(gaps) / len(gaps) / 6.0, 1.0)

        features[5 + min(obs.current_round, 3)] = 1.0

        opp_matrix = self._encode_action_matrix(self._opp_actions)
        self_matrix = self._encode_action_matrix(self._self_actions)
        for i in range(4):
            for j in range(4):
                features[9 + i * 4 + j] = opp_matrix[i][j] + 1.0
                features[25 + i * 4 + j] = self_matrix[i][j] + 1.0

        features[41] = 1.0 if len(obs.community_cards) > self._prev_community_count else 0.0

        call_amount = obs.current_bet - obs.player_round_bet
        total_after_call = obs.pot + call_amount
        if total_after_call > 0:
            features[42] = min(call_amount / total_after_call, 1.0)

        eff_stack = min(obs.player_chips, obs.opponent_chips)
        if obs.pot > 0:
            features[43] = min(eff_stack / obs.pot / 20.0, 1.0)

        features[44] = 0.5 if opp_equity is None else opp_equity
        features[45] = 0.5 if opp_rank_avg is None else opp_rank_avg
        features[46] = 0.5 if opp_suited is None else opp_suited
        features[47] = obs.betting_level / 6.0
        features[48] = float(obs.position)
        features[49] = max(0, MAX_RAISES - obs.raises_this_round) / 5.0
        # HU: dealer acts first preflop, second postflop
        is_ip = ((obs.position != obs.dealer_pos) if obs.current_round == 0
                 else obs.position == obs.dealer_pos)
        features[50] = 1.0 if is_ip else 0.0
        features[51] = min(eff_stack / 2000.0, 1.0)
        features[52] = len(obs.legal_actions) / 3.0
        return features

    def _encode_action_matrix(self, action_history: list) -> list:
        matrix = [[-1.0] * 4 for _ in range(4)]
        round_actions = defaultdict(list)
        for r, a in action_history:
            round_actions[r].append(a)
        for r in range(4):
            for i, a in enumerate(round_actions.get(r, [])):
                if i < 4:
                    matrix[r][i] = float(a)
        return matrix
