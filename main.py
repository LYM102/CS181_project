# main.py - Minimalist Texas Hold'em external entry point

"""
Usage examples:
    # Run random Agent match
    python main.py --num_hands 100

    # Specify different Agents for match
    python main.py --agent0 random --agent1 random --num_hands 1000
"""

import argparse
from collections import defaultdict

from game.engine import GameEngine
from game.constants import ACTION_NAMES, ROUND_NAMES
from game.card import cards_to_pretty
from agents.random_agent import RandomAgent
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SARSAAgent
from agents.bayesian_mc_agent import BayesianMCAgent
from agents.nn_mc_agent import NN_MCAgent


# ==================== Agent Registry ====================
AGENT_REGISTRY = {
    "random": RandomAgent,
    "expert": ExpertAgent,
    "sarsa": SARSAAgent,
    "bayesian_mc": BayesianMCAgent,
    "nn_mc": NN_MCAgent,
}


def create_agent(agent_type: str, player_id: int):
    """Create Agent instance from type string"""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}")
    cls = AGENT_REGISTRY[agent_type]
    return cls(name=f"{agent_type}_p{player_id}")


def run_interactive(num_hands: int = 1, verbose: bool = True):
    """Interactive single-hand run (with verbose output)"""
    agent0 = RandomAgent(name="Random_P0")
    agent1 = RandomAgent(name="Random_P1")
    engine = GameEngine(agent0, agent1)

    for hand_idx in range(num_hands):
        print(f"\n{'='*50}")
        print(f"  Hand #{hand_idx + 1}")
        print(f"{'='*50}")

        obs = engine.reset_hand()
        done = False

        if verbose:
            print(engine.display_state())

        while not done:
            player = engine.current_player
            action = engine.agents[player].act(obs)
            print(
                f"\n  Player {player} ({engine.agents[player].name}): {ACTION_NAMES[action]}")

            obs, reward, done, info = engine.step(action)

            if verbose:
                print(engine.display_state())

        # Display result
        result = info.get("result")
        if result:
            print(f"\n  --- Result ---")
            if result.winner is not None:
                print(
                    f"  Winner: Player {result.winner} ({engine.agents[result.winner].name})")
                print(f"  Hand: {result.hand_class}")
            else:
                print(f"  Tie! Pot split.")
            print(f"  Pot: {result.pot}")
            for pid in range(2):
                print(f"  Player {pid} reward: {result.rewards[pid]}")
                if pid in result.player_hands:
                    rank, cls = result.player_hands[pid]
                    print(f"  Player {pid} hand: {cls}")

    # Display final chips
    print(f"\n{'='*50}")
    print("  Final Chips:")
    for i in range(2):
        print(
            f"  Player {i} ({engine.agents[i].name}): {engine.players[i].chips}")


def run_evaluation(agent0_type: str, agent1_type: str, num_hands: int = 1000):
    """Batch evaluate match performance between two Agents"""
    agent0 = create_agent(agent0_type, 0)
    agent1 = create_agent(agent1_type, 1)
    engine = GameEngine(agent0, agent1)

    results = engine.run(num_hands=num_hands)

    # Statistics
    wins = defaultdict(int)
    ties = 0
    total_reward = defaultdict(float)

    for r in results:
        if r.winner is not None:
            wins[r.winner] += 1
        else:
            ties += 1
        for pid in range(2):
            total_reward[pid] += r.rewards[pid]

    print(f"\n{'='*60}")
    print(f"  Evaluation: {agent0.name} vs {agent1.name}")
    print(f"  Total hands: {num_hands}")
    print(f"{'='*60}")
    print(
        f"  Player 0 ({agent0.name}) wins: {wins[0]} ({wins[0]/num_hands*100:.1f}%)")
    print(
        f"  Player 1 ({agent1.name}) wins: {wins[1]} ({wins[1]/num_hands*100:.1f}%)")
    print(f"  Ties: {ties} ({ties/num_hands*100:.1f}%)")
    print(f"  Avg reward P0: {total_reward[0]/num_hands:.2f}")
    print(f"  Avg reward P1: {total_reward[1]/num_hands:.2f}")


def run_step_by_step():
    """Step-by-step execution mode (suitable for RL training debugging)"""
    agent0 = RandomAgent(name="Random_P0")
    agent1 = RandomAgent(name="Random_P1")
    engine = GameEngine(agent0, agent1)

    obs = engine.reset_hand()
    print("Initial observation:")
    print(f"  Hole cards: {obs.hole_cards_pretty}")
    print(f"  Community: {obs.community_cards_pretty}")
    print(f"  Pot: {obs.pot}, Current bet: {obs.current_bet}")
    print(f"  Legal actions: {[ACTION_NAMES[a] for a in obs.legal_actions]}")
    print(f"  Equity: {obs.equity:.4f}")
    print()

    done = False
    step = 0
    while not done:
        player = engine.current_player
        action = engine.agents[player].act(obs)
        obs, reward, done, info = engine.step(action)
        step += 1

        print(f"Step {step}: Player {player} → {ACTION_NAMES[action]}")
        print(
            f"  Pot: {obs.pot}, Round: {ROUND_NAMES.get(obs.current_round, '?')}")
        if done:
            result = info.get("result")
            if result:
                print(
                    f"  Hand over! Winner: Player {result.winner}, Reward: {result.rewards}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimalist Texas Hold'em AI Platform")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["interactive", "evaluate", "step"],
                        help="Run mode: interactive, evaluate (batch evaluation), step (step-by-step)")
    parser.add_argument("--agent0", type=str, default="random",
                        help="Player 0 Agent type")
    parser.add_argument("--agent1", type=str, default="random",
                        help="Player 1 Agent type")
    parser.add_argument("--num_hands", type=int, default=10,
                        help="Number of hands to play")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")

    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive(num_hands=args.num_hands, verbose=args.verbose)
    elif args.mode == "evaluate":
        run_evaluation(args.agent0, args.agent1, num_hands=args.num_hands)
    elif args.mode == "step":
        run_step_by_step()
