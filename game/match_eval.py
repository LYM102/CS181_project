# game/match_eval.py — Unified match evaluation (WR + zero-sum AvgR)
"""
AvgR = mean per-hand chip delta for the reported player.

Each HandResult.rewards[player] is chips_after_hand - chips_at_hand_start
(blinds included). Rewards are zero-sum: rewards[0] + rewards[1] == 0.

Evaluation protocol: reset both players to STARTING_CHIPS before every hand
so each hand is independent (no stack carryover / bankruptcy distortion).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from game.engine import GameEngine
from game.constants import STARTING_CHIPS, NUM_PLAYERS


AgentLike = Union[object, Callable[[], object]]


@dataclass
class MatchResult:
    """Aggregated match statistics for one agent vs one opponent."""
    num_hands: int
    report_agent_id: int
    wins: dict[int, int]
    ties: int
    total_reward: dict[int, float]
    win_rate: float          # percent for report agent
    avg_reward: float        # mean per-hand chip delta (report agent)
    zero_sum_residual: float # avg_r0 + avg_r1 (should be ~0)


def _resolve_agent(agent: AgentLike):
    return agent() if callable(agent) else agent


def reset_eval_stacks(engine: GameEngine) -> None:
    """Reset both players to STARTING_CHIPS before an independent eval hand."""
    if engine.players:
        for p in engine.players:
            p.chips = STARTING_CHIPS


def _reset_stacks(engine: GameEngine) -> None:
    reset_eval_stacks(engine)


def run_match(
    agent0: AgentLike,
    agent1: AgentLike,
    num_hands: int = 1000,
    seed: int | None = None,
    report_agent_id: int = 0,
) -> MatchResult:
    """
    Play num_hands independent heads-up hands and aggregate WR / AvgR.

    Args:
        agent0: player-0 agent instance or factory
        agent1: player-1 agent instance or factory
        num_hands: number of hands
        seed: optional RNG seed (random + numpy)
        report_agent_id: player id for win_rate / avg_reward (default 0)

    Returns:
        MatchResult with zero-sum per-hand averaged rewards.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    a0 = _resolve_agent(agent0)
    a1 = _resolve_agent(agent1)
    engine = GameEngine(a0, a1)

    wins = defaultdict(int)
    ties = 0
    total_reward = defaultdict(float)

    for _ in range(num_hands):
        _reset_stacks(engine)
        result = engine.run_hand()
        if result.winner is None:
            ties += 1
        else:
            wins[result.winner] += 1
        for pid in range(NUM_PLAYERS):
            total_reward[pid] += result.rewards[pid]

    avg_r = {pid: total_reward[pid] / num_hands for pid in range(NUM_PLAYERS)}
    wr = wins[report_agent_id] / num_hands * 100.0

    return MatchResult(
        num_hands=num_hands,
        report_agent_id=report_agent_id,
        wins=dict(wins),
        ties=ties,
        total_reward=dict(total_reward),
        win_rate=wr,
        avg_reward=avg_r[report_agent_id],
        zero_sum_residual=avg_r[0] + avg_r[1],
    )
