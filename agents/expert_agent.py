"""CFR-based expert (equilibrium) agent."""

from __future__ import annotations
import random
import os

from agents.base_agent import BaseAgent
from game.engine import Observation
from game.constants import FOLD, CALL, RAISE
from game.cfr_solver import CFRSolver

_DEFAULT_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cfr_policy.pkl"
)


class ExpertAgent(BaseAgent):
    """Nash equilibrium agent using external-sampling MCCFR."""

    def __init__(
        self,
        name: str = "ExpertAgent",
        policy_path: str | None = None,
        train_iterations: int = 0,
    ):
        super().__init__(name=name)
        self.policy_path = policy_path or _DEFAULT_POLICY_PATH
        self.solver = CFRSolver()
        self._policy_loaded = False

        if os.path.exists(self.policy_path):
            self.solver.load_policy(self.policy_path)
            self._policy_loaded = True
        elif train_iterations > 0:
            print(
                f"[ExpertAgent] Training CFR for {train_iterations} iterations...")
            self.solver.train(iterations=train_iterations)
            self.solver.save_policy(self.policy_path)
            self._policy_loaded = True

    def act(self, obs: Observation) -> int:
        """Sample action from CFR strategy table; fall back to equity heuristic."""
        legal_actions = obs.legal_actions

        if self._policy_loaded:
            probs = self.solver.get_action_prob(
                player=obs.position,
                hole_cards=obs.hole_cards,
                community_cards=obs.community_cards,
                betting_round=obs.current_round,
                betting_level=obs.betting_level,
                raises_this_round=obs.raises_this_round,
                legal_actions=legal_actions,
            )

            r = random.random()
            cum = 0.0
            for a in legal_actions:
                cum += probs[a]
                if r <= cum:
                    return a
            return legal_actions[-1]

        return self._heuristic_act(obs)

    def get_action_probs(self, obs: Observation) -> list[float]:
        """Return CFR mixed strategy [P(FOLD), P(CALL), P(RAISE)] for distillation."""
        if self._policy_loaded:
            return self.solver.get_action_prob(
                player=obs.position,
                hole_cards=obs.hole_cards,
                community_cards=obs.community_cards,
                betting_round=obs.current_round,
                betting_level=obs.betting_level,
                raises_this_round=obs.raises_this_round,
                legal_actions=obs.legal_actions,
            )
        return self._heuristic_probs(obs)

    def _heuristic_probs(self, obs: Observation) -> list[float]:
        """Equity-based fallback mixed strategy."""
        legal = obs.legal_actions
        probs = [0.0, 0.0, 0.0]
        if obs.equity > 0.65:
            if RAISE in legal:
                probs[RAISE] = 0.65; probs[CALL] = 0.30; probs[FOLD] = 0.05
            else:
                probs[CALL] = 0.85; probs[FOLD] = 0.15
        elif obs.equity > 0.40:
            probs[CALL] = 0.65; probs[RAISE] = 0.20 if RAISE in legal else 0.0; probs[FOLD] = 0.15
            if RAISE not in legal:
                probs = [0.20, 0.80, 0.0]
        elif obs.equity > 0.25:
            probs[FOLD] = 0.40; probs[CALL] = 0.50; probs[RAISE] = 0.10 if RAISE in legal else 0.0
            if RAISE not in legal:
                probs = [0.45, 0.55, 0.0]
        else:
            probs[FOLD] = 0.75; probs[CALL] = 0.20; probs[RAISE] = 0.05 if RAISE in legal else 0.0
            if RAISE not in legal:
                probs = [0.80, 0.20, 0.0]
        total = sum(probs[a] for a in legal)
        if total > 0:
            for a in range(3):
                if a in legal:
                    probs[a] /= total
                else:
                    probs[a] = 0.0
        return probs

    def _heuristic_act(self, obs: Observation) -> int:
        """Equity-based fallback action."""
        legal = obs.legal_actions

        if obs.equity > 0.65:
            if RAISE in legal:
                return RAISE
            return CALL
        elif obs.equity > 0.4:
            return CALL
        elif obs.equity > 0.25:
            if obs.current_bet <= 10:
                return CALL
            return FOLD
        else:
            return FOLD

    def train(self, iterations: int = 50000, save: bool = True) -> None:
        self.solver.train(iterations=iterations)
        self._policy_loaded = True
        if save:
            self.solver.save_policy(self.policy_path)
