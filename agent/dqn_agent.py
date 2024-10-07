import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQNAgent:
    def __init__(self, env, num_time_bins=5, hidden_size=128, learning_rate=1e-3, gamma=0.99):
        self.env = env
        self.num_time_bins = num_time_bins
        self.gamma = gamma
        
        # Discretize action space
        self.action_list = self._discretize_actions()
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}
        self.action_size = len(self.action_list)
        
        # Define state size
        self.state_size = self._get_state_size()
        
        # Initialize networks
        self.policy_net = DQN(self.state_size, self.action_size, hidden_size)
        self.target_net = DQN(self.state_size, self.action_size, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def _discretize_actions(self):
        action_list = []
        for loc in range(self.env.m + 1):  # Including Home
            time_lower = self.env.T_data_lower[loc]
            time_upper = self.env.T_data_upper[loc]
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=self.num_time_bins)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
    
    def _get_state_size(self):
        # Size of state vector
        # current_location (1) + remaining_time (1) + visited (m+1) + weather (1)
        return 1 + 1 + (self.env.m + 1) + 2
    
    def state_to_tensor(self, state):
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        visited = state['visited']
        weather = state['weather']
        
        # Normalize remaining_time
        remaining_time /= self.env.T_max
        
        # Convert weather to one-hot encoding
        weather_one_hot = np.zeros(2)
        weather_one_hot[weather] = 1
        
        # Construct state vector
        state_vector = np.concatenate((
            np.array([current_location / (self.env.m + 1)]),
            np.array([remaining_time]),
            visited,
            weather_one_hot
        ))
        return torch.FloatTensor(state_vector)
    
    def get_valid_action_mask(self, state):
        mask = np.zeros(len(self.action_list), dtype=bool)
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        visited = state['visited']
        weather = state['weather']
        
        for idx, (loc, t_data) in self.action_index_mapping.items():
            # Check if location has been visited
            if loc != 0 and visited[loc] == 1:
                continue
            # Check data collection time bounds
            if not (self.env.T_data_lower[loc] <= t_data <= self.env.T_data_upper[loc]):
                continue
            # Check remaining time constraints
            T_flight_to_next = self.env._get_flight_time(current_location, loc, weather)
            T_return = self.env._expected_return_time(loc)
            total_time_needed = T_flight_to_next + t_data + T_return
            if remaining_time < total_time_needed:
                continue
            mask[idx] = True
        return mask
    
    def select_action(self, state, epsilon):
        valid_actions = np.where(self.get_valid_action_mask(state))[0]
        if len(valid_actions) == 0:
            return None  # No valid actions
        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                q_values = q_values.detach().numpy()
                # Mask invalid actions
                invalid_actions = np.setdiff1d(np.arange(len(self.action_list)), valid_actions)
                q_values[invalid_actions] = -np.inf
                return np.argmax(q_values)
    
    def optimize_model(self, batch):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        
        batch_state_tensor = torch.stack([self.state_to_tensor(s) for s in batch_state])
        batch_action_tensor = torch.LongTensor(batch_action)
        batch_reward_tensor = torch.FloatTensor(batch_reward)
        batch_next_state_tensor = torch.stack([self.state_to_tensor(s) for s in batch_next_state])
        batch_done_tensor = torch.FloatTensor(batch_done)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(batch_state_tensor)
        state_action_values = q_values.gather(1, batch_action_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_state_tensor)
            next_state_values = torch.zeros(len(batch_next_state))
            for idx, next_state in enumerate(batch_next_state):
                mask = self.get_valid_action_mask(next_state)
                if mask.any():
                    next_q_values[idx][~torch.tensor(mask)] = -float('inf')
                    next_state_values[idx] = next_q_values[idx].max()
                else:
                    next_state_values[idx] = 0.0  # No valid actions
            expected_state_action_values = batch_reward_tensor + (self.gamma * next_state_values * (1 - batch_done_tensor))
        
        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
