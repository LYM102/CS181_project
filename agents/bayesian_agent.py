# agents/bayesian_agent.py
"""
BayesianAgent: Bayesian opponent modeling + tabular SARSA Q-table.

Win-rate oriented version:
    - Bayesian opponent belief is included in Q-table state.
    - Supports winrate_no_fold mode:
        if CALL/RAISE exists, FOLD is removed from action candidates.
      This is useful when the goal is purely maximizing hand win rate.
    - Uses heuristic fallback for unseen/tied Q states.
"""

from __future__ import annotations

import random
import pickle
from collections import defaultdict

from agents.base_agent import BaseAgent
from game.engine import Observation
from game.evaluator import equity_to_bin, pot_to_bin
from game.constants import FOLD, CALL, RAISE

from agents.bayesian_model import (
    BayesianLikelihoodEstimator,
    BayesianOpponentModel,
    WEAK,
    MID,
    STRONG,
)


class BayesianAgent(BaseAgent):
    """
    Bayesian opponent model + tabular SARSA agent.
    """

    ACTION_SPACE = [FOLD, CALL, RAISE]

    def __init__(
        self,
        name: str = "BayesianAgent",
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9998,
        epsilon_min: float = 0.02,
        alpha: float = 0.05,
        gamma: float = 1.0,
        player_id: int = 0,
        likelihood_path: str | None = None,
        load_model_path: str | None = None,
        explore_avoid_fold: bool = True,
        use_heuristic_fallback: bool = True,
        winrate_no_fold: bool = False,
    ):
        super().__init__(name=name)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma
        self.player_id = player_id

        # Win-rate oriented switches.
        self.explore_avoid_fold = explore_avoid_fold
        self.use_heuristic_fallback = use_heuristic_fallback
        self.winrate_no_fold = winrate_no_fold

        # Q-table: state tuple -> [Q_fold, Q_call, Q_raise]
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])

        if likelihood_path:
            self.likelihood = BayesianLikelihoodEstimator.load(likelihood_path)
        else:
            self.likelihood = BayesianLikelihoodEstimator()

        self.opponent_model = BayesianOpponentModel(self.likelihood)

        if load_model_path:
            self.load_model(load_model_path)

    # ============================================================
    # BaseAgent interface
    # ============================================================

    def act(self, obs: Observation) -> int:
        """
        Epsilon-greedy action selection.

        If winrate_no_fold=True:
            - Remove FOLD from candidate actions whenever CALL or RAISE is legal.
            - This is useful when the evaluation target is hand win rate,
              because FOLD guarantees losing the hand.
        """
        state = self._encode_state(obs)
        legal_actions = list(obs.legal_actions)

        # Candidate action set.
        decision_actions = legal_actions
        if self.winrate_no_fold:
            non_fold_actions = [a for a in legal_actions if a != FOLD]
            if non_fold_actions:
                decision_actions = non_fold_actions

        # Exploration.
        if random.random() < self.epsilon:
            return self._exploration_action(decision_actions)

        # Check before defaultdict creates state.
        state_seen = state in self.q_table

        q_vals = self.q_table[state]

        legal_q_values = [q_vals[a] for a in decision_actions]
        best_value = max(legal_q_values)
        worst_value = min(legal_q_values)

        # If unseen state or all legal Q-values are tied, use heuristic.
        if self.use_heuristic_fallback and (
            (not state_seen) or abs(best_value - worst_value) < 1e-12
        ):
            return self._heuristic_action(obs, decision_actions)

        best_actions = [a for a in decision_actions if q_vals[a] == best_value]

        # Tie-break by heuristic.
        if self.use_heuristic_fallback and len(best_actions) > 1:
            h_action = self._heuristic_action(obs, best_actions)
            if h_action in best_actions:
                return h_action

        return random.choice(best_actions)

    def _exploration_action(self, legal_actions: list[int]) -> int:
        """
        Exploration policy.

        Avoid random folding if possible.
        """
        if self.explore_avoid_fold:
            non_fold_actions = [a for a in legal_actions if a != FOLD]
            if non_fold_actions:
                return random.choice(non_fold_actions)

        return random.choice(legal_actions)

    def _heuristic_action(self, obs: Observation, legal_actions: list[int]) -> int:
        """
        Fallback policy for unseen/tied Q states.

        Designed for win-rate:
            - Strong equity -> raise.
            - Decent equity -> call.
            - Opponent weak + decent equity -> raise.
            - Very weak hand under high pressure against strong opponent -> fold,
              unless winrate_no_fold has removed FOLD from legal_actions.
        """
        equity = obs.equity
        betting_level = obs.betting_level
        opp_belief = self._get_belief_label_with_mid_tie_break()

        # Strong hand: raise.
        if RAISE in legal_actions and equity >= 0.70:
            return RAISE

        # Opponent likely weak, we can be aggressive.
        if RAISE in legal_actions and opp_belief == WEAK and equity >= 0.50:
            return RAISE

        # Very weak against strong opponent under pressure.
        if (
            FOLD in legal_actions
            and opp_belief == STRONG
            and equity < 0.22
            and betting_level >= 2
        ):
            return FOLD

        # Prefer call over fold for win rate.
        if CALL in legal_actions:
            if equity >= 0.18:
                return CALL

            if betting_level <= 1:
                return CALL

        # Medium hand and call unavailable.
        if RAISE in legal_actions and equity >= 0.55:
            return RAISE

        if FOLD in legal_actions:
            return FOLD

        return random.choice(legal_actions)

    def reset(self) -> None:
        """
        Reset per-hand Bayesian posterior.
        """
        self.opponent_model.reset()

    def update(
        self,
        obs: Observation,
        action: int,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        """
        Generic update hook. Not used here because SARSA needs next_action.
        """
        pass

    # ============================================================
    # Bayesian opponent action recording
    # ============================================================

    def record_action(
        self,
        player: int,
        action: int,
        round_num: int | None = None,
        bet_level: int | None = None,
    ) -> None:
        """
        Record opponent action and update posterior.
        """
        if player == self.player_id:
            return

        if bet_level is None:
            bet_level = 0

        self.opponent_model.update(action, bet_level)

    # ============================================================
    # State encoding
    # ============================================================

    def _encode_state(self, obs: Observation) -> tuple:
        """
        State:
            (own_equity_bin, betting_level, pot_bin, opponent_belief)

        opponent_belief:
            0 = weak
            1 = mid
            2 = strong
        """
        h_code = equity_to_bin(obs.equity, bins=20)
        pot_bin = pot_to_bin(obs.pot)
        opp_belief = self._get_belief_label_with_mid_tie_break()

        return (
            h_code,
            obs.betting_level,
            pot_bin,
            opp_belief,
        )

    def _get_belief_label_with_mid_tie_break(self) -> int:
        """
        If posterior is uniform, return MID instead of WEAK.
        """
        probs = self.opponent_model.get_probs()

        if max(probs) - min(probs) < 1e-9:
            return MID

        return self.opponent_model.get_label()

    def get_belief_probs(self) -> list[float]:
        return self.opponent_model.get_probs()

    def get_belief_label(self) -> int:
        return self._get_belief_label_with_mid_tie_break()

    # ============================================================
    # SARSA learning
    # ============================================================

    def learn_sarsa(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple | None,
        next_action: int | None,
        done: bool,
    ) -> None:
        """
        SARSA update.
        """
        q_current = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            q_next = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * q_next

        td_error = td_target - q_current
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_exploration(self, epsilon: float = 1.0) -> None:
        self.epsilon = epsilon

    # ============================================================
    # Utilities
    # ============================================================

    def get_q_table_size(self) -> int:
        return len(self.q_table)

    def save_model(self, filepath: str) -> None:
        """
        Save Q-table and Bayesian likelihood.
        """
        data = {
            "name": self.name,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "player_id": self.player_id,
            "explore_avoid_fold": self.explore_avoid_fold,
            "use_heuristic_fallback": self.use_heuristic_fallback,
            "winrate_no_fold": self.winrate_no_fold,
            "q_table": dict(self.q_table),
            "likelihood": {
                "num_buckets": self.likelihood.num_buckets,
                "num_actions": self.likelihood.num_actions,
                "num_bet_levels": self.likelihood.num_bet_levels,
                "laplace": self.likelihood.laplace,
                "counts": dict(self.likelihood.counts),
                "totals": dict(self.likelihood.totals),
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(
            f"BayesianAgent model saved to {filepath} "
            f"(Qsize={len(self.q_table)}, "
            f"epsilon={self.epsilon:.4f}, "
            f"heuristic_fallback={self.use_heuristic_fallback}, "
            f"explore_avoid_fold={self.explore_avoid_fold}, "
            f"winrate_no_fold={self.winrate_no_fold}, "
            f"likelihood_records={self.likelihood.total_records()})"
        )

    def load_model(self, filepath: str) -> None:
        """
        Load Q-table and Bayesian likelihood.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.epsilon_min = data.get("epsilon_min", self.epsilon_min)
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.player_id = data.get("player_id", self.player_id)

        self.explore_avoid_fold = data.get(
            "explore_avoid_fold",
            self.explore_avoid_fold,
        )
        self.use_heuristic_fallback = data.get(
            "use_heuristic_fallback",
            self.use_heuristic_fallback,
        )
        self.winrate_no_fold = data.get(
            "winrate_no_fold",
            getattr(self, "winrate_no_fold", False),
        )

        self.q_table.clear()
        for state, q_vals in data.get("q_table", {}).items():
            self.q_table[state] = q_vals

        if "likelihood" in data:
            ldata = data["likelihood"]
            self.likelihood = BayesianLikelihoodEstimator(
                num_buckets=ldata["num_buckets"],
                num_actions=ldata["num_actions"],
                num_bet_levels=ldata["num_bet_levels"],
                laplace=ldata["laplace"],
            )
            self.likelihood.counts.update(ldata["counts"])
            self.likelihood.totals.update(ldata["totals"])
            self.opponent_model = BayesianOpponentModel(self.likelihood)

        print(
            f"BayesianAgent model loaded from {filepath} "
            f"(Qsize={len(self.q_table)}, "
            f"epsilon={self.epsilon:.4f}, "
            f"heuristic_fallback={self.use_heuristic_fallback}, "
            f"explore_avoid_fold={self.explore_avoid_fold}, "
            f"winrate_no_fold={self.winrate_no_fold}, "
            f"likelihood_records={self.likelihood.total_records()})"
        )