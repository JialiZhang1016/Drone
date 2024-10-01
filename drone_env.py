import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneRoutePlanningEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(DroneRoutePlanningEnv, self).__init__()
        
        # Load configuration parameters
        self.m = config['num_locations']  # Number of locations excluding Home
        self.T_max = config['T_max']      # Total maximum fly time
        self.p = config['weather_prob']   # Probability of good weather
        self.P_penalty = config['P_penalty']  # Penalty for not returning home
        
        # Flight time matrices
        self.T_flight_good = np.array(config['T_flight_good'])
        self.T_flight_bad = np.array(config['T_flight_bad'])
        
        # Data collection time bounds
        self.T_data_lower = np.array(config['T_data_lower'])
        self.T_data_upper = np.array(config['T_data_upper'])
        
        # Criticality levels
        self.criticality = config['criticality']  # List of 'HC' or 'LC' for each location
        
        # Define action and observation spaces
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.m + 1),  # Next location (including Home)
            spaces.Box(low=self.T_data_lower, high=self.T_data_upper, dtype=np.float32)  # Data collection time
        ))
        
        # Observation space: L_t, T_t_rem, V_t (one-hot), W_t
        self.observation_space = spaces.Dict({
            'current_location': spaces.Discrete(self.m + 1),
            'remaining_time': spaces.Box(low=0, high=self.T_max, shape=(1,), dtype=np.float32),
            'visited': spaces.MultiBinary(self.m + 1),
            'weather': spaces.Discrete(2)  # 0: Good, 1: Bad
        })
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # This line resets the RNG of the environment
        
        self.L_t = 0  # Start at Home
        self.T_t_rem = self.T_max
        self.V_t = np.zeros(self.m + 1, dtype=np.int8)  # Visited locations
        self.V_t[0] = 1  # Mark Home as visited
        self.W_t = np.random.choice([0, 1], p=[self.p, 1 - self.p])  # 0: Good, 1: Bad
        
        self.visit_order = [0]  # Initialize visit order with starting location

        self.done = False
        self.total_reward = 0.0
        self.timestep = 0
        
        return self._get_observation(), {}  # Return observation and an empty info dict
    
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has finished. Call reset() to start a new episode.")
        
        # Check safety factor: if remaining time <= 20% T_max and not at Home, force return Home
        if self.T_t_rem <= 0.2 * self.T_max and self.L_t != 0:
            L_next = 0
            T_data_next = 0.0
        else:
            L_next, T_data_next = action
            L_next = int(L_next)
            T_data_next = float(T_data_next)
        
            # If all locations are visited, force return Home
            if np.all(self.V_t[1:] == 1) and L_next != 0:
                L_next = 0
                T_data_next = 0.0
        
        # Validate action constraints
        if not self._is_valid_action(L_next, T_data_next):
            # Invalid action, apply a large negative reward
            reward = -self.P_penalty
            self.done = True
            return self._get_observation(), reward, self.done, {}
        
        # Weather update
        self.W_t = np.random.choice([0, 1], p=[self.p, 1 - self.p])  # 0: Good, 1: Bad
        
        # Flight time to next location
        T_flight_to_next = self._get_flight_time(self.L_t, L_next, self.W_t)
        
        # Update remaining fly time
        self.T_t_rem -= (T_flight_to_next + T_data_next)
        
        # Update current location
        self.L_t = L_next
        
        # Update visited locations
        self.V_t[L_next] = 1

        # Update visit order
        self.visit_order.append(L_next)
        
        observation = self._get_observation()
        # Calculate reward
        reward = self._calculate_reward(L_next, T_data_next, T_flight_to_next)
        self.total_reward += reward
        
        # Check if episode is done
        self.done = self._check_done()
        
        # Apply penalty at the end if not at Home
        if self.done and self.L_t != 0:
            reward -= self.P_penalty
            self.total_reward -= self.P_penalty
    
        self.timestep += 1
        
        info = {
            'action': {
                'next_location': int(L_next),
                'data_collection_time': round(float(T_data_next), 2)
            },
            'next_state': {
                'current_location': int(self.L_t),
                'remaining_time': round(float(self.T_t_rem), 2),
                'visited': self.V_t.tolist(),
                'weather': 'Good' if self.W_t == 0 else 'Bad'
            },
            'reward': round(float(reward), 2),
            'total_reward': round(float(self.total_reward), 2),
            'visited_locations': self.visit_order
        }
        truncated = False
        return observation, reward, self.done, info
    
    def _get_observation(self):
        observation = {
            'current_location': self.L_t,
            'remaining_time': np.array([self.T_t_rem], dtype=np.float32),
            'visited': self.V_t.copy(),
            'weather': self.W_t
        }
        return observation
    
    def _is_valid_action(self, L_next, T_data_next):
        # Check if next location is valid
        if L_next != 0 and self.V_t[L_next] == 1:
            return False  # Already visited
        
        # Check data collection time bounds
        if not (self.T_data_lower[L_next] <= T_data_next <= self.T_data_upper[L_next]):
            return False
        
        # Flight time to next location
        T_flight_to_next = self._get_flight_time(self.L_t, L_next, self.W_t)
        
        # Expected return time to Home
        T_return = self._expected_return_time(L_next)
        
        # Check remaining fly time constraint
        total_time_needed = T_flight_to_next + T_data_next + T_return
        if self.T_t_rem < total_time_needed:
            return False
        
        return True
    
    def _get_flight_time(self, L_from, L_to, weather):
        if weather == 0:  # Good
            return self.T_flight_good[L_from, L_to]
        else:  # Bad
            return self.T_flight_bad[L_from, L_to]
    
    def _expected_return_time(self, L_from):
        # Expected flight time from L_from to Home
        T_return_good = self.T_flight_good[L_from, 0]
        T_return_bad = self.T_flight_bad[L_from, 0]
        T_expected_return = self.p * T_return_good + (1 - self.p) * T_return_bad
        return T_expected_return
    
    def _calculate_reward(self, L_next, T_data_next, T_flight_to_next):
        # Data collection reward
        if self.criticality[L_next] == 'HC':
            R_data = 10 * T_data_next
        else:
            R_data = 2 * T_data_next
        
        # Cost
        C = -1 * (T_data_next + T_flight_to_next)
        
        return R_data + C
    
    def _check_done(self):
        # Episode ends if:
        # 1. Drone returns to Home
        # 2. No remaining fly time
        # 3. Cannot proceed due to constraints
        if self.L_t == 0 and self.timestep > 0:
            return True
        if self.T_t_rem <= 0:
            return True
        if np.all(self.V_t == 1):
            return True
        return False

    def render(self, mode='human'):   
        print(f"Current Location: {self.L_t}")
        print(f"Remaining Time: {self.T_t_rem}")
        print(f"Visited Locations: {self.visit_order}")
        print(f"Weather: {'Good' if self.W_t == 0 else 'Bad'}")
        print(f"Step Reward: {self.reward}")
        print(f"Total Reward: {self.total_reward}")
        print("-----")

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
        current_location = observation['current_location']
        remaining_time = observation['remaining_time'][0]
        visited = observation['visited']
        weather = observation['weather']

        # Get all unvisited locations, including Home (location number 0)
        unvisited = np.where(np.logical_or(visited == 0, np.arange(len(visited)) == 0))[0]

        # Decide the next location
        if len(unvisited) == 1 or remaining_time <= 0:
            # Only Home is left or out of time
            next_location = 0
        else:
            # Randomly select an unvisited location
            next_location = np.random.choice(unvisited)

        # Determine the data collection time based on the selected location
        T_data_min = self.env.T_data_lower[next_location]
        T_data_max = self.env.T_data_upper[next_location]

        if T_data_min == 0 and T_data_max == 0:
            # If selecting to return Home, data collection time is 0
            T_data_next = 0.0
        else:
            # Randomly select a data collection time within the allowed range
            T_data_next = np.random.uniform(T_data_min, T_data_max)

        return (int(next_location), float(T_data_next))

