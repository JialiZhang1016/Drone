import matplotlib.pyplot as plt
import numpy as np

def linear_epsilon_decay(episode, max_episodes, min_epsilon=0.01):
    """
    Linear decay of ε from 1 to min_epsilon, decay process lasts for 0.8 * max_episodes.
    
    Parameters:
        episode (int): Current training episode.
        max_episodes (int): Maximum number of training episodes.
        min_epsilon (float, optional): Minimum ε value. Default is 0.01.
        
    Returns:
        float: Current ε value.
    """
    #return max(min_epsilon, 1 - episode / (0.8 * max_episodes))
    return max(min_epsilon, np.exp(-0.00005*episode))

def exponential_epsilon_decay(episode, epsilon_start, epsilon_end, decay_rate):
    """
    Exponential decay of ε from epsilon_start to epsilon_end, decay rate controlled by decay_rate.
    
    Parameters:
        episode (int): Current training episode.
        epsilon_start (float): Initial ε value.
        epsilon_end (float): Final ε value.
        decay_rate (float): Decay rate.
        
    Returns:
        float: Current ε value.
    """
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / decay_rate)

def plot_epsilon_decay(total_episodes, linear_params_list, exponential_params_list):
    """
    Plot curves for linear decay and exponential decay ε strategies.
    
    Parameters:
        total_episodes (int): Total number of episodes to consider for plotting.
        linear_params_list (list of dict): List of parameters for linear decay strategy, each dict contains 'max_episodes' and 'min_epsilon'.
        exponential_params_list (list of dict): List of parameters for exponential decay strategy, each dict contains 'epsilon_start', 'epsilon_end' and 'decay_rate'.
    """
    episodes = np.arange(1, total_episodes + 1)
    plt.figure(figsize=(14, 8))
    
    # Plot linear decay curves
    for params in linear_params_list:
        max_eps = params.get('max_episodes', 1000)
        min_eps = params.get('min_epsilon', 0.01)
        epsilons = [linear_epsilon_decay(ep, max_eps, min_eps) for ep in episodes]
        label = f"Linear decay: max_episodes={max_eps}, min_epsilon={min_eps}"
        plt.plot(episodes, epsilons, label=label, linestyle='-', linewidth=2)
    
    # Plot exponential decay curves
    for params in exponential_params_list:
        epsilon_start = params.get('epsilon_start', 1.0)
        epsilon_end = params.get('epsilon_end', 0.01)
        decay_rate = params.get('decay_rate', 500)
        epsilons = [exponential_epsilon_decay(ep, epsilon_start, epsilon_end, decay_rate) for ep in episodes]
        label = f"Exponential decay: start={epsilon_start}, end={epsilon_end}, decay_rate={decay_rate}"
        plt.plot(episodes, epsilons, label=label, linestyle='--', linewidth=2)
    
    plt.title('Comparison of ε Decay Strategies', fontsize=18)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Epsilon (ε)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define total number of training episodes
    total_episodes = 300000
    
    # Define different parameter settings for linear decay strategy
    linear_params_list = [
        {'max_episodes': 300000, 'min_epsilon': 0.01},
        # {'max_episodes': 8000, 'min_epsilon': 0.01},
        # {'max_episodes': 6000, 'min_epsilon': 0.01}
    ]
    
    # Define different parameter settings for exponential decay strategy
    exponential_params_list = [
        {'epsilon_start': 0.4, 'epsilon_end': 0.001, 'decay_rate': 300}
        # {'epsilon_start': 1.0, 'epsilon_end': 0.05, 'decay_rate': 3000},
        # {'epsilon_start': 1.0, 'epsilon_end': 0.1, 'decay_rate': 1000}
    ]
    
    # Plot ε decay curves
    plot_epsilon_decay(total_episodes, linear_params_list, exponential_params_list=exponential_params_list)
