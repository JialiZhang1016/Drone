import gymnasium as gym
from drone_env import DroneRoutePlanningEnv
from agent.dqn_agent import DQNAgent
import json
import torch
import numpy as np
import random
import sys
import os

def evaluate_dqn(
    config_file: str,
    model_path: str,
    num_episodes: int = 1,
    seed: int = 42,
    verbose: bool = True,
    use_action_mask: bool = True
) -> list:
    """
    Evaluate a trained DQN agent on the DroneRoutePlanningEnv.

    Parameters:
        config_file (str): Path to the configuration JSON file.
        model_path (str): Path to the trained policy network parameters (.pth file).
        num_episodes (int, optional): Number of episodes to evaluate. Defaults to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        verbose (bool, optional): If True, prints actions and episode results. Defaults to True.

    Returns:
        List[dict]: A list of dictionaries containing episode results.
                    Each dictionary includes 'episode', 'total_reward', and 'steps'.
    """

    # Load configuration file
    try:
        with open(config_file, 'r') as config_file_handle:
            config = json.load(config_file_handle)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the configuration file: {config_file}")
        return []

    # Create environment
    try:
        env = DroneRoutePlanningEnv(config)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return []

    # Initialize agent
    try:
        agent = DQNAgent(env, use_action_mask=use_action_mask)
    except Exception as e:
        print(f"Error initializing DQNAgent: {e}")
        return []

    # Load trained model parameters
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        return []

    try:
        agent.policy_net.load_state_dict(torch.load(model_path,weights_only=True))
        agent.policy_net.eval()
    except Exception as e:
        print(f"Error loading model parameters: {e}")
        return []

    # Initialize results list
    results = []

    # Evaluation loop
    for episode in range(1, num_episodes + 1):
        try:
            state, _ = env.reset(seed=seed)
        except Exception as e:
            print(f"Error resetting environment: {e}")
            break

        total_reward = 0
        steps = 0
        done = False

        while not done:
            with torch.no_grad():
                action_idx = agent.select_action(state, epsilon=0.0)
                if action_idx is None:
                    if verbose:
                        print(f"No valid action available. Ending Episode {episode}.")
                    break  # No valid action, end episode
                action = agent.action_index_mapping[action_idx]
                if verbose:
                    print(f"Episode {episode}, Step {steps + 1}: Action Taken: {action}")
                try:
                    next_state, reward, done, info = env.step(action)
                except Exception as e:
                    print(f"Error during environment step: {e}")
                    break
                total_reward += reward
                state = next_state
                steps += 1

        if verbose:
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.2f}, Steps: {steps}, Seed: {seed}")

        # Append episode results
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps
        })

    if verbose:
        print("Evaluation completed.")

    return results


if __name__ == "__main__":

    evaluation_results = evaluate_dqn(
        # config_file="config/config_5_0.4.json",
        # model_path="runs/2024-10-26_19:09:20_5_6000/policy_net.pth",

        # config_file="config/config_10.json",
        # model_path="runs/2024-10-26_19:29:04_10_6000/policy_net.pth",

        config_file="config/realword_8/config_8.json",
        model_path="runs/2025-07-24_16:55:56_wp_0.6_8_2000/policy_net_final.pth",

        # config_file="config/updated_configs/config_20_wp_0.2.json",
        # model_path="outputs/2024-11-06_20:46:12_wp_0.2_20_5000/policy_net.pth",



        num_episodes=10,
        seed = 42,
        verbose=True,
        use_action_mask=True
    )

    num_eval_episodes = 10
    if evaluation_results:
        avg_reward = np.mean([res['total_reward'] for res in evaluation_results])
        avg_steps = np.mean([res['steps'] for res in evaluation_results])
        print(f"\nAverage Reward over {num_eval_episodes} Episodes: {avg_reward:.2f}")
        print(f"Average Steps over {num_eval_episodes} Episodes: {avg_steps:.2f}")
