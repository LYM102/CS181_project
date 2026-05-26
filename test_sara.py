from agents.sarsa_agent import SarsaAgent
from agents.random_agent import RandomAgent
from game.engine import GameEngine

agent = SarsaAgent(load_q_table_path="train/sarsa_phase1.pkl")
agent.epsilon = 0.0   # 贪婪
opponent = RandomAgent()
env = GameEngine(agent, opponent)

rewards = []
for _ in range(100):
    obs = env.reset_hand()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            rewards.append(reward)
print(f"Average reward over 100 hands: {sum(rewards)/len(rewards):.2f}")