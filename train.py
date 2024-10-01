import gymnasium as gym
from drone_env import DroneRoutePlanningEnv
from dqn_agent import DQNAgent
import json
import torch
import random
import numpy as np
from collections import deque

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file) 

# Create the environment
env = DroneRoutePlanningEnv(config)

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)

# Initialize agent
agent = DQNAgent(env)

# Hyperparameters
num_episodes = 5000
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 300
target_update_freq = 10
memory_size = 10000
gamma = 0.99

memory = deque(maxlen=memory_size)
steps_done = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
    
    for t in range(100):  # Limit max steps per episode
        steps_done += 1
        action_idx = agent.select_action(state, epsilon)
        if action_idx is None:
            break  # No valid actions, end episode
        action = agent.action_index_mapping[action_idx]
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Store experience
        memory.append((state, action_idx, reward, next_state, done))
        
        state = next_state
        
        # Experience replay
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            agent.optimize_model(batch)
        
        if done:
            break
    
    # Update target network
    if episode % target_update_freq == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

print("Training completed.")
