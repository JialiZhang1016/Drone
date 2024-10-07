import numpy as np

class RandomAgent:
    """
    An agent that selects a random action from the environment.
    """

    def __init__(self, env):
        """
        Initialize the random agent.
        """
        self.env = env

    def select_action(self, observation):
        """
        Select a valid random action based on the current observation.
        """
        _, remaining_time, visited, _ = observation
        
        # Choose a random unvisited location or return home
        unvisited = np.where(visited == 0)[0]
        if len(unvisited) == 0 or remaining_time <= 0:
            next_location = 0  # Return home
        else:
            next_location = np.random.choice(unvisited)
        
        # Choose a random data collection time within bounds
        T_data_min = self.env.T_data_lower[next_location]
        T_data_max = self.env.T_data_upper[next_location]
        T_data_next = np.random.uniform(T_data_min, T_data_max)
        
        return (next_location, T_data_next)

