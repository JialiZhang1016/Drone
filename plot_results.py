import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# python plot_results.py runs/5_500_20241007-004833 

# Get the results directory path
if len(sys.argv) > 1:
    results_dir = sys.argv[1]
else:
    print("Please provide the results directory path, e.g.: python plot_analysis.py runs/3_500_20231007-123456")
    sys.exit(1)

# Load data
all_rewards = np.load(os.path.join(results_dir, 'all_rewards.npy'))
all_losses = np.load(os.path.join(results_dir, 'all_losses.npy'))
episode_lengths = np.load(os.path.join(results_dir, 'episode_lengths.npy'))
epsilons = np.load(os.path.join(results_dir, 'epsilons.npy'))
success_rates = np.load(os.path.join(results_dir, 'success_rates.npy'))
average_rewards = np.load(os.path.join(results_dir, 'average_rewards.npy'))
moving_average_rewards = np.load(os.path.join(results_dir, 'moving_average_rewards.npy'))

# Plot and save images
def plot_and_save(data, ylabel, filename, x=None):
    plt.figure()
    if x is None:
        plt.plot(data)
    else:
        plt.plot(x, data)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.savefig(os.path.join(results_dir, f"{filename}.png"))
    plt.close()

# Plot all rewards
plot_and_save(all_rewards, 'Total Reward per Episode', 'all_rewards')

# Plot all losses
plot_and_save(all_losses, 'Average Loss per Episode', 'all_losses')

# Plot episode lengths
plot_and_save(episode_lengths, 'Episode Length', 'episode_lengths')

# Plot epsilon values
plot_and_save(epsilons, 'Epsilon', 'epsilons')

# Plot success rates
success_rate_interval = 10  # Keep consistent with the training code
success_rate_episodes = np.arange(success_rate_interval, len(success_rates)*success_rate_interval+1, success_rate_interval)
plot_and_save(success_rates, 'Success Rate', 'success_rates', x=success_rate_episodes)

# Plot moving average rewards
moving_average_interval = 20  # Keep consistent with the training code
moving_avg_reward_episodes = np.arange(moving_average_interval-1, len(moving_average_rewards)+moving_average_interval-1)
plot_and_save(moving_average_rewards, 'Moving Average Reward', 'moving_average_rewards', x=moving_avg_reward_episodes)

print("All charts have been saved in the directory:", results_dir)
