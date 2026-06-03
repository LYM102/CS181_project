# agents/expert_agent.py - Nash equilibrium expert strategy based on custom CFR (52-card)

from __future__ import annotations
import random
import os

from agents.base_agent import BaseAgent
from game.engine import Observation
from game.constants import FOLD, CALL, RAISE
from game.cfr_solver import CFRSolver


# Default policy file path
_DEFAULT_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cfr_policy.pkl"
)


class ExpertAgent(BaseAgent):
    """
    Nash equilibrium baseline Agent based on custom CFR (Counterfactual Regret Minimization).

    Key implementation details:
      - Uses External Sampling MCCFR to solve approximate Nash equilibrium
        for standard 52-card heads-up Texas Hold'em
      - Trained policy cached to file, loaded on subsequent runs
      - Queries action probability distribution from strategy table based on current information set
        (hole_bucket, community_bucket, round, bet_level, raises), then samples by probability

    Usage:
        # First use: train and save
        agent = ExpertAgent(train_iterations=50000)

        # Subsequent use: auto-load from file
        agent = ExpertAgent()
    """

    def __init__(
        self,
        name: str = "ExpertAgent",
        policy_path: str | None = None,
        train_iterations: int = 0,
    ):
        """
        Args:
            name: Agent name
            policy_path: policy file path, None uses default path
            train_iterations: if > 0 and policy file doesn't exist, train for specified iterations
        """
        super().__init__(name=name)
        self.policy_path = policy_path or _DEFAULT_POLICY_PATH
        self.solver = CFRSolver()
        self._policy_loaded = False

        # Try to load existing policy
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
        """
        Select action based on CFR policy.

        Samples by probability from the strategy table's probability distribution,
        falls back to equity-based heuristic if info set was never seen.
        """
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

            # Sample by probability
            r = random.random()
            cum = 0.0
            for a in legal_actions:
                cum += probs[a]
                if r <= cum:
                    return a
            return legal_actions[-1]

        # No policy: equity-based heuristic
        return self._heuristic_act(obs)

    def _heuristic_act(self, obs: Observation) -> int:
        """Equity-based heuristic strategy (fallback when no CFR policy)"""
        legal = obs.legal_actions

        if obs.equity > 0.65:
            # Strong hand: prefer raise
            if RAISE in legal:
                return RAISE
            return CALL
        elif obs.equity > 0.4:
            # Medium hand: mostly call
            return CALL
        elif obs.equity > 0.25:
            # Weak hand: depends on situation
            if obs.current_bet <= 10:
                return CALL
            return FOLD
        else:
            # Very weak: fold
            return FOLD

    def train(self, iterations: int = 50000, save: bool = True) -> None:
        """
        Manually trigger CFR training.

        Args:
            iterations: number of training iterations
            save: whether to save policy after training
        """
        self.solver.train(iterations=iterations)
        self._policy_loaded = True
        if save:
            self.solver.save_policy(self.policy_path)
