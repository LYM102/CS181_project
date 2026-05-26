# train_sarsa.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pickle
from game.engine import GameEngine
from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent
from agents.expert_agent import ExpertAgent

def train_one_hand(env, agent):
    """训练一手牌，返回净收益"""
    obs = env.reset_hand()
    state = agent._encode_state(obs)
    action = agent.act(obs)
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        step_count += 1
        if step_count > 50:
            print("Warning: step limit reached, breaking")
            break
        next_obs, reward, done, _ = env.step(action)
        if done:
            agent.learn(state, action, reward, None, None, done=True)
            total_reward = reward
            break
        next_action = agent.act(next_obs)
        next_state = agent._encode_state(next_obs)
        agent.learn(state, action, reward, next_state, next_action, done=False)
        state, action = next_state, next_action

    agent.decay_epsilon()
    return total_reward

def evaluate_agent(agent, opponent, num_hands=1000):
    """评估智能体平均净收益（epsilon设为0）"""
    env = GameEngine(agent, opponent)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    total_reward = 0.0
    for _ in range(num_hands):
        obs = env.reset_hand()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                total_reward += reward
    agent.epsilon = original_epsilon
    return total_reward / num_hands

def train():
    # 创建 SARSA 智能体
    agent = SarsaAgent(name="Sarsa", alpha=0.1, gamma=0.95,
                       epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01)
    
    # 第一阶段：对抗随机对手
    print("Phase 1: Training vs RandomAgent...")
    opponent = RandomAgent()
    env = GameEngine(agent, opponent)
    for hand in range(1, 100001):
        train_one_hand(env, agent)
        if hand % 100 == 0:
            print(f"{hand} ", end="", flush=True)
        if hand % 5000 == 0:
            avg_reward = evaluate_agent(agent, RandomAgent(), 500)
            print(f"Hand {hand} | ε={agent.epsilon:.4f} | Qsize={agent.get_q_table_size()} | "
                  f"AvgReward vs Random: {avg_reward:.2f}")

    agent.save_q_table("sarsa_phase1.pkl")

    # 第二阶段：对抗专家对手
    print("Phase 2: Training vs ExpertAgent...")
    opponent = ExpertAgent()
    env = GameEngine(agent, opponent)
    for hand in range(1, 400001):
        train_one_hand(env, agent)
        if hand % 10000 == 0:
            avg_reward = evaluate_agent(agent, RandomAgent(), 500)
            print(f"Hand {hand} | ε={agent.epsilon:.4f} | Qsize={agent.get_q_table_size()} | "
                  f"AvgReward vs Random: {avg_reward:.2f}")
    
    agent.save_q_table("sarsa_final.pkl")
    print("Training completed. Final Q-table saved.")

if __name__ == "__main__":
    train()