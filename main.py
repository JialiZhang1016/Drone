import json
import numpy as np
import os
from drone_env import DroneRoutePlanningEnv
import contextlib

def main():
    # Define output directory
    output_dir = 'outputs'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the next available file number
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('env_test') and f.endswith('.txt')]
    next_number = max([int(f.split('_')[2].split('.')[0]) for f in existing_files], default=0) + 1
    output_filename = os.path.join(output_dir, f'env_test_{next_number}.txt')
    
    # Open output file and redirect stdout
    with open(output_filename, 'w') as f:
        with contextlib.redirect_stdout(f):
            # Load configuration from JSON file
            with open('config.json', 'r') as config_file:
                config = json.load(config_file)
            
            # Create environment
            env = DroneRoutePlanningEnv(config)
            
            max_episodes = 5  # You can adjust this number

            for episode in range(max_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                
                print(f"Episode {episode + 1}")
                print("=" * 30)
                
                while not done:
                    # Get current observation
                    current_location = state['current_location']
                    remaining_time = state['remaining_time'][0]
                    visited = state['visited']
                    weather = state['weather']
                    
                    # Choose a random unvisited location or return home
                    unvisited = np.where(visited == 0)[0]
                    if len(unvisited) == 0 or remaining_time <= 0:
                        next_location = 0  # Return home
                    else:
                        next_location = np.random.choice(unvisited)
                    
                    # Choose a random data collection time within bounds
                    T_data_min = env.T_data_lower[next_location]
                    T_data_max = env.T_data_upper[next_location]
                    T_data_next = np.random.uniform(T_data_min, T_data_max)
                    
                    action = (next_location, T_data_next)
                    
                    # Take action in the environment
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    # Render current state
                    env.render()
                    
                    # Print action and reward information
                    print(f"Action: Move to Location {next_location}, Data Collection Time: {T_data_next:.2f}")
                    print(f"Reward: {reward:.2f}")
                    print(f"Done: {done}")
                    print("-----")

                # Print episode summary
                print(f"Episode {episode + 1} finished.")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Final Location: {env.L_t}")
                print(f"Visited Locations: {env.visit_order}")
                print("\n" + "=" * 50 + "\n")

    print(f"Results have been written to {output_filename}")

if __name__ == "__main__":
    main()
