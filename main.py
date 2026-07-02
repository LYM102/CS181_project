# main.py



import argparse
from collections import defaultdict

from game.engine import GameEngine
from game.constants import ACTION_NAMES, ROUND_NAMES
from game.card import cards_to_pretty
from agents.random_agent import RandomAgent
from agents.expert_agent import ExpertAgent
from agents.sarsa_agent import SarsaAgent
from agents.l1_agent import L1Agent
from agents.l2_agent import L2Agent
from agents.l3_agent import L3Agent


AGENT_REGISTRY = {
    "random": RandomAgent,
    "expert": ExpertAgent,
    "sarsa": SarsaAgent,
    "l1": L1Agent,
    "l2": L2Agent,
    "l3": L3Agent,
    "nn_mc": L1Agent,
}


def create_agent(agent_type: str, player_id: int, model_path: str = None):

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}")
    cls = AGENT_REGISTRY[agent_type]
    if agent_type == "sarsa" and model_path:
        return cls(name=f"{agent_type}_p{player_id}", load_q_table_path=model_path)
    if agent_type in ("l1", "l2", "nn_mc", "l3") and model_path:
        return cls(name=f"{agent_type}_p{player_id}", load_model_path=model_path)
    return cls(name=f"{agent_type}_p{player_id}")


def run_interactive(num_hands: int = 1, verbose: bool = True):

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


def run_evaluation(agent0_type: str, agent1_type: str, num_hands: int,
                   sarsa_model0: str = None, sarsa_model1: str = None,
                   belief_model0: str = None, belief_model1: str = None):

    def _get_model_path(atype, sarsa_p, belief_p):
        if atype == "sarsa":
            return sarsa_p
        if atype in ("l1", "l2", "nn_mc", "l3"):
            return belief_p
        return None

    model0 = _get_model_path(agent0_type, sarsa_model0, belief_model0)
    model1 = _get_model_path(agent1_type, sarsa_model1, belief_model1)
    agent0 = create_agent(agent0_type, 0, model_path=model0)
    agent1 = create_agent(agent1_type, 1, model_path=model1)

    if agent0_type in ("sarsa", "l1", "l2", "nn_mc", "l3"):
        agent0.epsilon = 0.0
    if agent1_type in ("sarsa", "l1", "l2", "nn_mc", "l3"):
        agent1.epsilon = 0.0

    if agent0_type in ("l1", "l2", "nn_mc") and hasattr(agent0, "_auto_record_self"):
        agent0._auto_record_self = False
    if agent1_type in ("l1", "l2", "nn_mc") and hasattr(agent1, "_auto_record_self"):
        agent1._auto_record_self = False

    from game.match_eval import run_match
    stats = run_match(agent0, agent1, num_hands=num_hands, report_agent_id=0)
    wins = stats.wins
    ties = stats.ties
    avg_r0 = stats.avg_reward
    avg_r1 = stats.total_reward[1] / num_hands

    print(f"\n{'='*60}")
    print(f"  Evaluation: {agent0.name} vs {agent1.name}")
    print(f"  Total hands: {num_hands}")
    print(f"{'='*60}")
    print(
        f"  Player 0 ({agent0.name}) wins: {wins.get(0, 0)} ({wins.get(0, 0)/num_hands*100:.1f}%)")
    print(
        f"  Player 1 ({agent1.name}) wins: {wins.get(1, 0)} ({wins.get(1, 0)/num_hands*100:.1f}%)")
    print(f"  Ties: {ties} ({ties/num_hands*100:.1f}%)")
    print(f"{'='*60}")
    print(f"  --- Chip-Based Metrics (zero-sum) ---")
    print(f"  Total chips P0: {stats.total_reward[0]:+.0f}  |  Avg chips/hand: {avg_r0:+.2f}")
    print(f"  Total chips P1: {stats.total_reward[1]:+.0f}  |  Avg chips/hand: {avg_r1:+.2f}")
    print(f"  Zero-sum check: AvgR sum = {stats.zero_sum_residual:+.4f}")
    if avg_r0 > avg_r1:
        print(f"  → P0 dominates: +{avg_r0 - avg_r1:.2f} chips/hand advantage")
    elif avg_r1 > avg_r0:
        print(f"  → P1 dominates: +{avg_r1 - avg_r0:.2f} chips/hand advantage")


def run_step_by_step():

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
        description="Standard Texas Hold'em AI Platform (52-card)")
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
    parser.add_argument("--sarsa_model0", type=str, default=None,
                        help="Path to saved Q-table for SARSA agent 0")
    parser.add_argument("--sarsa_model1", type=str, default=None,
                        help="Path to saved Q-table for SARSA agent 1")
    parser.add_argument("--belief_model0", type=str, default=None,
                        help="Path to saved model for L1/L2/L3 agent 0")
    parser.add_argument("--belief_model1", type=str, default=None,
                        help="Path to saved model for L1/L2/L3 agent 1")

    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive(num_hands=args.num_hands, verbose=args.verbose)
    elif args.mode == "evaluate":
        run_evaluation(args.agent0, args.agent1, args.num_hands,
                   args.sarsa_model0, args.sarsa_model1,
                   args.belief_model0, args.belief_model1)
    elif args.mode == "step":
        run_step_by_step()
