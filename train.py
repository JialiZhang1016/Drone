# train.py
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
import csv  # Import csv module for writing CSV files
from plot_results import plot_results  # Ensure this function is correctly implemented
from evaluate import evaluate_dqn  # Import the evaluation function

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
    learning_rate=1e-3,
    num_time_bins=5,
    hidden_size=128,
    seed=42,
    success_rate_interval=100,
    moving_average_interval=100,
    save_interval=100,
    results_base_dir='runs',
    use_action_mask=True  # Added parameter to control action mask usage
) -> str:
    """
    Train a DQN agent for drone route planning.

    Parameters:
        ... [Parameters as previously defined]
        use_action_mask (bool): Whether to use the action mask during training.

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
    
    # Initialize agent with use_action_mask parameter
    agent = DQNAgent(env, 
                     use_action_mask=use_action_mask, 
                     gamma=gamma, 
                     learning_rate=learning_rate, 
                     num_time_bins=num_time_bins, 
                     hidden_size=hidden_size)
    
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
        f"{current_time}_wp_{config['weather_prob']}_{num_locations}_{num_episodes}"
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
        "learning_rate": learning_rate,
        "num_time_bins": num_time_bins,
        "hidden_size": hidden_size,
        "seed": seed,
        "success_rate_interval": success_rate_interval,
        "moving_average_interval": moving_average_interval,
        "save_interval": save_interval,
        "results_base_dir": results_base_dir,
        "use_action_mask": use_action_mask,  # Include the use_action_mask parameter
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

def update_config(original_config_path, new_weather_prob, output_config_path):
    """
    Reads the original config file, updates the weather_prob, and writes to a new config file.

    Parameters:
        original_config_path (str): Path to the original config file.
        new_weather_prob (float): New weather probability to set.
        output_config_path (str): Path to save the updated config file.
    """
    with open(original_config_path, 'r') as f:
        config = json.load(f)
    
    config['weather_prob'] = new_weather_prob
    
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated config saved to {output_config_path} with weather_prob={new_weather_prob}")

# Removed the save_evaluation_results function as it's no longer needed

if __name__ == "__main__":
    # Original configuration file
    original_config_file = 'config/config_5.json'
    
    # Weather probabilities to iterate over
    weather_probs = [0.2, 0.5, 0.8, 1.0]
    # Training parameters
    num_episodes = 1000
    use_action_mask = True
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 300
    target_update_freq = 50
    memory_size = 1000
    gamma = 0.99
    num_time_bins = 5
    hidden_size = 128
    seed = 42
    success_rate_interval = 50
    moving_average_interval = 50
    save_interval = 1000
    results_base_dir = 'runs'
    updated_config_dir = 'config/updated_configs'  # Directory to save updated config files
    
    # Ensure the results and updated config directories exist
    os.makedirs(results_base_dir, exist_ok=True)
    os.makedirs(updated_config_dir, exist_ok=True)
    
    # Initialize a list to store summary data
    summary_data = []
    
    for wp in weather_probs:
        # Define a unique config filename for each weather_prob
        config_filename = f"config_5_wp_{wp}.json"
        config_path = os.path.join(updated_config_dir, config_filename)
            
        # Update the config with the new weather_prob
        update_config(original_config_file, wp, config_path)
                
        # Train the DQN agent with the updated config and use_action_mask setting
        results_dir = train_dqn(
            config_file=config_path,
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
            results_base_dir=results_base_dir,
            use_action_mask=use_action_mask  # Pass the use_action_mask parameter
        )
                
        # Plot the results after training
        plot_results(results_dir=results_dir)
                
        # Evaluate the trained model
        evaluation_results = evaluate_dqn(
            config_file=config_path,
            model_path=os.path.join(results_dir, 'policy_net.pth'),
            num_episodes=1000,  # Adjust as needed
            seed=seed,
            verbose=False,  # Set to True if you want detailed logs
            use_action_mask=use_action_mask  # Pass the use_action_mask parameter if needed
        )
                
        # Calculate average reward and average steps
        avg_reward = np.mean([res['total_reward'] for res in evaluation_results])
        avg_steps = np.mean([res['steps'] for res in evaluation_results])
                
        # Append the results to summary_data
        summary_data.append({
            'weather_prob': wp,
            'use_action_mask': use_action_mask,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps
        })
                
        # Optionally, print the summary for each weather_prob and use_action_mask setting
        mask_status = "With Mask" if use_action_mask else "Without Mask"
        print(f"Weather Prob: {wp}, {mask_status}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}\n")

    
    # After all trainings, save the summary_data to a CSV file
    summary_csv_path = os.path.join(results_base_dir, 'summary_results.csv')
    with open(summary_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['weather_prob', 'use_action_mask', 'avg_reward', 'avg_steps']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for data in summary_data:
            writer.writerow(data)
    
    print(f"Summary of results saved to {summary_csv_path}")
