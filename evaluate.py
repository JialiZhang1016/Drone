import gymnasium as gym
from drone_env import DroneRoutePlanningEnv
from agent.dqn_agent import DQNAgent
import json
import torch
import numpy as np

# 加载配置文件
with open('config/config_5.json', 'r') as config_file:
    config = json.load(config_file) 

# 创建环境
env = DroneRoutePlanningEnv(config)

# 初始化智能体
agent = DQNAgent(env)

# 加载训练好的模型参数
agent.policy_net.load_state_dict(torch.load('runs/5_500_20241007-003429/policy_net.pth'))
agent.policy_net.eval()

num_episodes = 10  # 评估 10 个 Episode

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        # 在评估时，不使用 epsilon 贪心策略，而是直接选择 Q 值最大的动作
        with torch.no_grad():
            action_idx = agent.select_action(state, epsilon=0.0)
            if action_idx is None:
                break  # 无有效动作，结束 Episode
            action = agent.action_index_mapping[action_idx]
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
    
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Steps: {steps}")

print("Evaluation completed.")
