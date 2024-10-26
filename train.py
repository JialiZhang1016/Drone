import gymnasium as gym
from drone_env import DroneRoutePlanningEnv
from agent.dqn_agent import DQNAgent
import json
import torch
import random
import numpy as np
from collections import deque
import os
from datetime import datetime

def train_dqn(
    config_file='config/config_5.json',
    num_episodes=5000,
    batch_size=32,
    epsilon_start=1.0,
    epsilon_end=0.001,
    epsilon_decay=300,
    target_update_freq=10,
    memory_size=10000,
    gamma=0.99,
    seed=42,
    success_rate_interval=10,
    moving_average_interval=20,
    save_interval=100,
    results_base_dir='runs'
):
    """
    Train a DQN agent for drone route planning.

    Parameters:
        config_file (str): Path to the configuration JSON file.
        num_episodes (int): Number of training episodes.
        batch_size (int): Size of the minibatch for experience replay.
        epsilon_start (float): Starting value of epsilon for epsilon-greedy policy.
        epsilon_end (float): Minimum value of epsilon.
        epsilon_decay (int): Decay rate for epsilon.
        target_update_freq (int): Frequency (in episodes) to update the target network.
        memory_size (int): Maximum size of the replay memory.
        gamma (float): Discount factor for future rewards.
        seed (int): Random seed for reproducibility.
        success_rate_interval (int): Number of episodes over which to calculate success rate.
        moving_average_interval (int): Window size for moving average of rewards.
        save_interval (int): Interval (in episodes) to print progress.
        results_base_dir (str): Base directory to save results.

    Returns:
        None
    """
    # Load configuration file
    with open(config_file, 'r') as config_file_handle:
        config = json.load(config_file_handle) 
    
    # Create environment
    env = DroneRoutePlanningEnv(config)
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    
    # Initialize agent
    agent = DQNAgent(env)
    
    # Initialize replay memory
    memory = deque(maxlen=memory_size)
    steps_done = 0
    
    # Get num_locations from config
    num_locations = config.get('num_locations', 0)  # Not including 'home'
    
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    # Create directory for saving results
    results_dir = os.path.join(
        results_base_dir,
        f"{current_time}_{num_locations}_{num_episodes}"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize metric lists
    all_rewards = []
    episode_lengths = []
    epsilons = []
    successes = []
    success_rates = []
    moving_average_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
        epsilons.append(epsilon)
        steps = 0  # Record steps for each episode
    
        while True:
            steps_done += 1
            steps += 1
            action_idx = agent.select_action(state, epsilon)
            if action_idx is None:
                break  # No valid action, end episode
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
        if (episode + 1) % target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
        all_rewards.append(total_reward)
        episode_lengths.append(steps)
    
        # Determine if successful (based on total_reward)
        success = total_reward >= 0
        successes.append(success)
    
        # Calculate success rate
        if (episode + 1) % success_rate_interval == 0:
            recent_successes = successes[-success_rate_interval:]
            success_rate = sum(recent_successes) / len(recent_successes)
            success_rates.append(success_rate)
    
        # Calculate moving average reward
        if len(all_rewards) >= moving_average_interval:
            recent_rewards = all_rewards[-moving_average_interval:]
            moving_avg_reward = sum(recent_rewards) / len(recent_rewards)
            moving_average_rewards.append(moving_avg_reward)
    
        if (episode + 1) % save_interval == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")
    
    # Save model parameters
    torch.save(agent.policy_net.state_dict(), os.path.join(results_dir, 'policy_net.pth'))
    
    # Save metric data
    np.save(os.path.join(results_dir, 'all_rewards.npy'), np.array(all_rewards))
    np.save(os.path.join(results_dir, 'episode_lengths.npy'), np.array(episode_lengths))
    np.save(os.path.join(results_dir, 'epsilons.npy'), np.array(epsilons))
    np.save(os.path.join(results_dir, 'success_rates.npy'), np.array(success_rates))
    np.save(os.path.join(results_dir, 'moving_average_rewards.npy'), np.array(moving_average_rewards))
    
    print("Training completed.") 
    print(f"Files are saved in: {results_dir}")


if __name__ == "__main__":
    train_dqn(
        config_file='config/config_5.json',
        num_episodes=5000,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.001,
        epsilon_decay=300,
        target_update_freq=10,
        memory_size=10000,
        gamma=0.99,
        seed=42
)
