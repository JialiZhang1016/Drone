import gymnasium as gym
import numpy as np
import json
from drone_env import DroneRoutePlanningEnv, RandomAgent
# Wrapper to discretize the continuous action space
class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, num_bins=5):
        super(DiscretizeActionWrapper, self).__init__(env)
        self.num_bins = num_bins  # Number of bins to discretize the data collection time
        self.m = self.env.m  # Number of locations excluding Home
        self.action_mapping = []
        
        # Build action mapping
        for L_next in range(self.m + 1):  # Locations from 0 to m
            T_data_lower = self.env.T_data_lower[L_next]
            T_data_upper = self.env.T_data_upper[L_next]
            if T_data_lower == T_data_upper:
                # Only one possible data collection time
                T_data_values = [T_data_lower]
            else:
                T_data_values = np.linspace(T_data_lower, T_data_upper, num_bins)
            for T_data_next in T_data_values:
                self.action_mapping.append((L_next, T_data_next))
        
        # Update the action space to be discrete
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))
    
    def action(self, action_index):
        # Map the discrete action index to the original action
        return self.action_mapping[action_index]

# Wrapper to flatten the observation space
class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenObservationWrapper, self).__init__(env)
        # Calculate the size of the flattened observation
        self.m = self.env.m
        self.flat_obs_size = 1 + 1 + (self.m + 1) + 1  # current_location + remaining_time + visited + weather
        # Define the new observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.flat_obs_size,), dtype=np.float32)
    
    def observation(self, observation):
        return self._flatten_observation(observation)
    
    def _flatten_observation(self, observation):
        # Convert the observation dictionary to a flat numpy array
        current_location = np.array([observation['current_location']], dtype=np.float32)
        remaining_time = observation['remaining_time']
        visited = observation['visited'].astype(np.float32)
        weather = np.array([observation['weather']], dtype=np.float32)
        return np.concatenate([current_location, remaining_time, visited, weather])
    
def DroneEnvWrapper(env, num_bins=5):
    env = DiscretizeActionWrapper(env, num_bins=num_bins)
    env = FlattenObservationWrapper(env)
    return env


"""
config = json.load(open('config.json', 'r'))
env = DroneRoutePlanningEnv(config)
env = DroneEnvWrapper(env, num_bins=5)

print(env.observation_space)
print(env.action_space)
"""

