import json
import os
from drone_env import DroneRoutePlanningEnv, RandomAgent
import contextlib

def main():
    # Define the output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Get the next available file number
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('env_test') and f.endswith('.txt')]
    next_number = max([int(f.split('_')[2].split('.')[0]) for f in existing_files], default=0) + 1
    output_filename = os.path.join(output_dir, f'env_test_{next_number}.txt')

    # Open the output file and redirect stdout
    with open(output_filename, 'w') as f:
        with contextlib.redirect_stdout(f):
            # Load configuration from JSON file
            with open('config.json', 'r') as config_file:
                config = json.load(config_file)

            # Create environment and agent instances
            env = DroneRoutePlanningEnv(config)
            agent = RandomAgent(env)

            max_episodes = 5  # Adjust this value as needed

            for episode in range(max_episodes):
                state = env.reset()
                done = False

                print(f"Episode {episode + 1}")
                print("=" * 30)

                while not done:
                    action = agent.select_action(state)
                    state, reward, done, info = env.step(action)
                    
                    # Print the info dictionary in a clear format
                    print("Action:")
                    print(f"  Next Location: {info['action']['next_location']}")
                    print(f"  Data Collection Time: {info['action']['data_collection_time']:.2f}")
                    print("Next State:")
                    print(f"  Current Location: {info['next_state']['current_location']}")
                    print(f"  Remaining Time: {info['next_state']['remaining_time']:.2f}")
                    print(f"  Visited: {info['next_state']['visited']}")
                    print(f"  Weather: {info['next_state']['weather']}")
                    print(f"Reward: {info['reward']:.2f}")
                    print(f"Total Reward: {info['total_reward']:.2f}")
                    print(f"Visited Locations: {info['visited_locations']}")
                    print("-----")

                episode_reward = env.total_reward
                # Print episode summary
                print("\n" + "=" * 50 + "\n")
                print(f"Episode {episode + 1} finished.")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Final Location: {env.L_t}")
                print(f"Visited Locations: {env.visit_order}")
                print("\n" + "=" * 50 + "\n")

    print(f"Results have been written to {output_filename}")

if __name__ == "__main__":
    main()
