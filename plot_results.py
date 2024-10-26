import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_results(
    results_dir,
    success_rate_interval=10,
    moving_average_interval=20,
):
    """
    Plot and save training metrics from the results directory.
    
    Parameters:
        results_dir (str): Path to the directory containing the result `.npy` files.
        success_rate_interval (int, optional): 
            Number of episodes over which to calculate the success rate.
            Defaults to 10.
        moving_average_interval (int, optional): 
            Window size for calculating the moving average of rewards.
            Defaults to 20.
    Returns:
        None
    """
    # Verify that the results directory exists
    if not os.path.isdir(results_dir):
        print(f"The specified results directory does not exist: {results_dir}")
        sys.exit(1)
    
    # Define the paths to the required `.npy` files
    required_files = [
        'all_rewards.npy',
        'episode_lengths.npy',
        'epsilons.npy',
        'success_rates.npy',
        'moving_average_rewards.npy'
    ]
    
    # Check for the existence of each required file
    for file_name in required_files:
        file_path = os.path.join(results_dir, file_name)
        if not os.path.isfile(file_path):
            print(f"Required file not found: {file_path}")
            sys.exit(1)
    
    # Load data
    all_rewards = np.load(os.path.join(results_dir, 'all_rewards.npy'))
    episode_lengths = np.load(os.path.join(results_dir, 'episode_lengths.npy'))
    epsilons = np.load(os.path.join(results_dir, 'epsilons.npy'))
    success_rates = np.load(os.path.join(results_dir, 'success_rates.npy'))
    moving_average_rewards = np.load(os.path.join(results_dir, 'moving_average_rewards.npy'))
    
    # Helper function to plot and save figures
    def plot_and_save(data, ylabel, filename, x=None, title=None):
        plt.figure()
        if x is None:
            plt.plot(data)
        else:
            plt.plot(x, data)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title if title else ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{filename}.png"))
        plt.close()
    
    # Plot all rewards
    plot_and_save(
        data=all_rewards,
        ylabel='Total Reward per Episode',
        filename='all_rewards',
        title='Total Reward per Episode'
    )
    
    # Plot episode lengths
    plot_and_save(
        data=episode_lengths,
        ylabel='Episode Length',
        filename='episode_lengths',
        title='Episode Length'
    )
    
    # Plot epsilon values
    plot_and_save(
        data=epsilons,
        ylabel='Epsilon',
        filename='epsilons',
        title='Epsilon over Episodes'
    )
    
    # Plot success rates
    success_rate_episodes = np.arange(
        success_rate_interval, 
        len(success_rates)*success_rate_interval + 1, 
        success_rate_interval
    )
    plot_and_save(
        data=success_rates,
        ylabel='Success Rate',
        filename='success_rates',
        x=success_rate_episodes,
        title='Success Rate over Episodes'
    )
    
    # Plot moving average rewards
    moving_avg_reward_episodes = np.arange(
        moving_average_interval - 1, 
        len(moving_average_rewards) + moving_average_interval - 1
    )
    plot_and_save(
        data=moving_average_rewards,
        ylabel='Moving Average Reward',
        filename='moving_average_rewards',
        x=moving_avg_reward_episodes,
        title='Moving Average Reward over Episodes'
    )
    
    print("All charts have been saved in the directory:", results_dir)


if __name__ == "__main__":   
    # Call the plot_results function with default intervals
    plot_results(results_dir="runs/5_5000_2024-10-25_19:12:52",
                 success_rate_interval=20,
                 moving_average_interval=20
                 )
