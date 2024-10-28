import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import List

def plot_results(
    results_dir,
):
    """
    Plot and save training metrics from the results directory.
    
    Parameters:
        results_dir (str): Path to the directory containing the result `.npy` files.
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
    plot_and_save(
        data=success_rates,
        ylabel='Success Rate',
        filename='success_rates',
        title='Success Rate over Episodes'
    )
    
    # Plot moving average rewards
    plot_and_save(
        data=moving_average_rewards,
        ylabel='Moving Average Reward',
        filename='moving_average_rewards',
        title='Moving Average Reward over Episodes'
    )
    
    print("All charts have been saved in the directory:", results_dir)




def plot_multiple_datasets_extended(
    data_paths: List[str],
    labels: List[str],
    title: str = 'Comparison of Moving Average Rewards',
    xlabel: str = 'Episode',
    ylabel: str = 'Moving Average Reward',
    save_path: str = None,
    show_plot: bool = True,
    apply_moving_average: bool = False,
    moving_average_interval: int = 100
) -> None:
    """
    Plot multiple datasets on a single graph for comparison, with an option to apply moving average.

    Parameters:
        data_paths (List[str]): List of paths to `.npy` data files.
        labels (List[str]): List of labels for each dataset.
        title (str, optional): Title of the plot. Defaults to 'Comparison of Moving Average Rewards'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Episode'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Moving Average Reward'.
        save_path (str, optional): Path to save the plot image. If None, the plot is not saved. Defaults to None.
        show_plot (bool, optional): If True, displays the plot. Defaults to True.
        apply_moving_average (bool, optional): If True, applies moving average to the data. Defaults to False.
        moving_average_interval (int, optional): The window size for moving average. Defaults to 100.

    Returns:
        None
    """
    if len(data_paths) != len(labels):
        raise ValueError("The number of data paths must match the number of labels.")

    plt.figure(figsize=(12, 8))

    for data_path, label in zip(data_paths, labels):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"The data file was not found: {data_path}")
        try:
            data = np.load(data_path)
        except Exception as e:
            raise ValueError(f"Error loading data from {data_path}: {e}")

        episodes = np.arange(1, len(data) + 1)

        if apply_moving_average:
            if moving_average_interval < 1:
                raise ValueError("moving_average_interval must be at least 1.")
            # Calculate moving average using a simple convolution
            window = np.ones(moving_average_interval) / moving_average_interval
            moving_avg = np.convolve(data, window, mode='valid')
            # Adjust episodes to match the moving average length
            ma_episodes = np.arange(moving_average_interval, len(data) + 1)
            plt.plot(ma_episodes, moving_avg, label=label, linewidth=2)
        else:
            plt.plot(episodes, data, label=label, linewidth=2)

    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close()

# # # extended plot
# if __name__ == "__main__":
#     data_files = [
#         'runs/q_learning_rewards_config_5_0.4.npy',
#         'runs/2024-10-26_19:09:20_5_6000/all_rewards.npy'
#     ]
#     labels = [
#         'Run 1: Q-Learning',
#         'Run 2: Our Method',
#     ]
#     plot_save_path = 'runs/m=5,p=0.4,T_max=400.png'
    
#     # Parameters for moving average
#     apply_ma = True
#     ma_interval = 100  # Adjust the window size as needed

#     plot_multiple_datasets_extended(
#         data_paths=data_files,
#         labels=labels,
#         title='m=20, p=0.4, T_max=1200',
#         xlabel='Episode',
#         ylabel='Moving Average Reward',
#         save_path=plot_save_path,
#         show_plot=True,
#         apply_moving_average=apply_ma,
#         moving_average_interval=ma_interval
#     )


# one plot
plot_results(results_dir="runs/2024-10-27_00:08:12_5_2000")

