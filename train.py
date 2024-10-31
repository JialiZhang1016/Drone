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
import time
from plot_results import plot_results  # Ensure this function is correctly implemented

def train_dqn(
    config_file='config/config_5.json',
    num_episodes=5000,
    batch_size=32,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=300,
    target_update_freq=10,
    memory_size=10000,
    gamma=0.99,
    seed=42,
    success_rate_interval=100,
    moving_average_interval=100,
    save_interval=100,
    results_base_dir='runs'
) -> str:
    """
    Train a DQN agent for drone route planning.

    Parameters:
        ... [Parameters as previously defined]

    Returns:
        results_dir (str): Directory where the training results are saved.
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
    
    # Save all configuration parameters
    all_configs = {
        "config_file": config_file,
        "num_episodes": num_episodes,
        "batch_size": batch_size,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "target_update_freq": target_update_freq,
        "memory_size": memory_size,
        "gamma": gamma,
        "seed": seed,
        "success_rate_interval": success_rate_interval,
        "moving_average_interval": moving_average_interval,
        "save_interval": save_interval,
        "results_base_dir": results_base_dir,
        "env_config": config  # Include the environment configuration
    }

    # Save all_configs to a JSON file in the results directory
    with open(os.path.join(results_dir, 'all_configs.json'), 'w') as f:
        json.dump(all_configs, f, indent=4)
    
    # Initialize metric lists
    all_rewards = []
    episode_lengths = []
    epsilons = []
    successes = []
    success_rates = []
    moving_average_rewards = []
    
    time_start = time.time()

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

    time_end = time.time()
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
    print(f"Time taken: {time_end - time_start:.2f} seconds")
    
    return results_dir  # Return the results directory path

if __name__ == "__main__":
    # Training parameters
    config_file = 'config/config_5_0.4.json'
    num_episodes = 2000
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 300
    target_update_freq = 50
    memory_size = 1000
    gamma = 0.99
    seed = 42
    success_rate_interval = 50
    moving_average_interval = 50
    save_interval = 1000
    results_base_dir = 'runs'

    # Train the DQN agent and capture the results directory
    results_dir = train_dqn(
        config_file=config_file,
        num_episodes=num_episodes,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        memory_size=memory_size,
        gamma=gamma,
        seed=seed,
        success_rate_interval=success_rate_interval,
        moving_average_interval=moving_average_interval,
        save_interval=save_interval,
        results_base_dir=results_base_dir
    )
    
    # Plot the results after training
    plot_results(results_dir=results_dir)
