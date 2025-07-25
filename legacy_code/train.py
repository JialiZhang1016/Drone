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
import csv
from plot_results import plot_results
from evaluate import evaluate_dqn

def train_dqn(
    config_file='config/realword_8/config_8_real.json',
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
    save_epochs=None,  # Changed from save_interval to save_epochs
    results_base_dir='runs',
    use_action_mask=True
) -> str:
    """
    Train a DQN agent for drone route planning.

    Parameters:
        ... [Parameters as previously defined]
        save_epochs (list): List of episode numbers at which to save the model.
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
        "save_epochs": save_epochs,  # Changed from save_interval
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

    if save_epochs is None:
        save_epochs = [num_episodes]

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

        # Save model at specified epochs
        if (episode + 1) in save_epochs:
            model_filename = f'policy_net_{episode + 1}.pth'
            torch.save(agent.policy_net.state_dict(), os.path.join(results_dir, model_filename))
            print(f"Model saved at episode {episode + 1}")

    time_end = time.time()
    # Save final model parameters
    torch.save(agent.policy_net.state_dict(), os.path.join(results_dir, 'policy_net_final.pth'))
    
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

if __name__ == "__main__":
    # Training parameters
    num_episodes = 2000
    save_epochs = [500,1000,2000]
    is_mask_list = [True]
    locations = ["config/realword_8/config_8.json"]
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 1000
    target_update_freq = 50
    memory_size = 1000
    gamma = 0.99
    num_time_bins = 5
    hidden_size = 128
    seed = 42
    success_rate_interval = 50
    moving_average_interval = 50

    # Initialize a list to store summary data
    summary_data = []

    for is_mask in is_mask_list:
        for config_file in locations:

            # Train the DQN agent
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
                save_epochs=save_epochs,
                use_action_mask=is_mask  # Pass the is_mask parameter
            )

            # Plot the results after training (optional)
            plot_results(results_dir=results_dir)

            # For each saved model at specified epochs
            for epoch in save_epochs:
                model_filename = f'policy_net_{epoch}.pth'
                model_path = os.path.join(results_dir, model_filename)

                # Evaluate the trained model
                evaluation_results = evaluate_dqn(
                    config_file=config_file,
                    model_path=model_path,
                    num_episodes=1000,  # Adjust as needed
                    seed=seed,
                    verbose=False,  # Set to True if you want detailed logs
                    use_action_mask=is_mask  # Pass the use_action_mask parameter
                )

                # Calculate average reward and average steps
                avg_reward = np.mean([res['total_reward'] for res in evaluation_results])
                avg_steps = np.mean([res['steps'] for res in evaluation_results])

                # Extract location number from config_file
                location_num = int(config_file.split('_')[-1].split('.')[0])

                # Append the results to summary_data
                summary_data.append({
                    'location': location_num,
                    'epoch': epoch,
                    'is_mask': 'TRUE' if is_mask else 'FALSE',
                    'avg_reward': avg_reward,
                    'avg_steps': avg_steps
                })

                # Optionally, print the summary
                mask_status = "TRUE" if is_mask else "FALSE"
                print(f"Location: {location_num}, Epoch: {epoch}, Is_Mask: {mask_status}, "
                      f"Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}")

    # After all trainings and evaluations, save the summary_data to a CSV file
    os.makedirs(os.path.join('runs', 'ablation_study'), exist_ok=True)
    summary_csv_path = os.path.join('runs/ablation_study', 'summary.csv')
    with open(summary_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['location', 'epoch', 'is_mask', 'avg_reward', 'avg_steps']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for data in summary_data:
            writer.writerow(data)

    print(f"Summary of results saved to {summary_csv_path}")
