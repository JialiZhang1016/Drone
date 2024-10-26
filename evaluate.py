import gymnasium as gym
from drone_env import DroneRoutePlanningEnv
from agent.dqn_agent import DQNAgent
import json
import torch
import numpy as np

# load config file
with open('config/config_5_real.json', 'r') as config_file:
    config = json.load(config_file) 

# Create environment
env = DroneRoutePlanningEnv(config)

# Initialize agent
agent = DQNAgent(env)

# Load trained model parameters
agent.policy_net.load_state_dict(torch.load('runs/5_5000_2024-10-25_18:57:23/policy_net.pth'))
agent.policy_net.eval()

num_episodes = 1  # Evaluate 10 episodes

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        # During evaluation, do not use epsilon-greedy strategy, but directly select the action with the highest Q value
        with torch.no_grad():
            action_idx = agent.select_action(state, epsilon=0.0)
            if action_idx is None:
                break  # No valid action, end episode
            action = agent.action_index_mapping[action_idx]
            print(action)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
    
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Steps: {steps}")

print("Evaluation completed.")
