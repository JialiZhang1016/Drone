#!/usr/bin/env python3
"""
Advanced Baseline Algorithms for Drone Route Planning

This module implements additional baseline algorithms for comparison:
- Double DQN (DDQN)
- Dueling DQN
- Shortest Path Algorithm (Dijkstra-based)
- Genetic Algorithm (GA)
- Simple Priority-based Algorithm

These serve as additional baselines to compare against the main DQN implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import heapq
from typing import List, Tuple, Dict, Optional
from collections import deque
import copy

class DoubleDQNAgent:
    """
    Double DQN implementation to reduce overestimation bias
    """
    
    def __init__(self, env, num_time_bins=5, hidden_size=128, learning_rate=1e-3, gamma=0.99):
        self.env = env
        self.num_time_bins = num_time_bins
        self.gamma = gamma
        
        # GPU device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Discretize action space
        self.action_list = self._discretize_actions()
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}
        self.action_size = len(self.action_list)
        
        # Define state size
        self.state_size = self._get_state_size()
        
        # Initialize networks
        self.policy_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
    
    def _discretize_actions(self):
        """Discretize action space"""
        action_list = []
        for loc in range(self.env.m + 1):
            time_lower, time_upper = self.env.get_data_time_bounds(loc)
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=self.num_time_bins)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
    
    def _get_state_size(self):
        """Get state vector size"""
        return 1 + 1 + (self.env.m + 1) + 3  # current_location + remaining_time + visited + weather
    
    def state_to_tensor(self, state):
        """Convert state to tensor"""
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        visited = state['visited']
        weather = state['weather']
        
        # Normalize remaining_time
        remaining_time /= self.env.T_max
        
        # Weather one-hot encoding
        weather_one_hot = np.zeros(3)
        weather_one_hot[weather] = 1
        
        # Construct state vector
        state_vector = np.concatenate((
            np.array([current_location / (self.env.m + 1)]),
            np.array([remaining_time]),
            visited,
            weather_one_hot
        ))
        return torch.FloatTensor(state_vector).to(self.device)
    
    def select_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
    
    def optimize_model(self, batch):
        """Optimize model using Double DQN update rule"""
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        
        batch_state_tensor = torch.stack([self.state_to_tensor(s) for s in batch_state])
        batch_action_tensor = torch.LongTensor(batch_action).to(self.device)
        batch_reward_tensor = torch.FloatTensor(batch_reward).to(self.device)
        batch_next_state_tensor = torch.stack([self.state_to_tensor(s) for s in batch_next_state])
        batch_done_tensor = torch.FloatTensor(batch_done).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(batch_state_tensor)
        state_action_values = q_values.gather(1, batch_action_tensor.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use policy network to select actions, target network to evaluate
        with torch.no_grad():
            next_q_values_policy = self.policy_net(batch_next_state_tensor)
            next_actions = torch.argmax(next_q_values_policy, dim=1)
            
            next_q_values_target = self.target_net(batch_next_state_tensor)
            next_state_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        expected_state_action_values = batch_reward_tensor + (self.gamma * next_state_values * (1 - batch_done_tensor))
        
        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DuelingDQNAgent:
    """
    Dueling DQN implementation with separate value and advantage streams
    """
    
    def __init__(self, env, num_time_bins=5, hidden_size=128, learning_rate=1e-3, gamma=0.99):
        self.env = env
        self.num_time_bins = num_time_bins
        self.gamma = gamma
        
        # GPU device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Discretize action space
        self.action_list = self._discretize_actions()
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}
        self.action_size = len(self.action_list)
        
        # Define state size
        self.state_size = self._get_state_size()
        
        # Initialize networks
        self.policy_net = DuelingDQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = DuelingDQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
    
    def _discretize_actions(self):
        """Discretize action space"""
        action_list = []
        for loc in range(self.env.m + 1):
            time_lower, time_upper = self.env.get_data_time_bounds(loc)
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=self.num_time_bins)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
    
    def _get_state_size(self):
        """Get state vector size"""
        return 1 + 1 + (self.env.m + 1) + 3
    
    def state_to_tensor(self, state):
        """Convert state to tensor"""
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        visited = state['visited']
        weather = state['weather']
        
        # Normalize remaining_time
        remaining_time /= self.env.T_max
        
        # Weather one-hot encoding
        weather_one_hot = np.zeros(3)
        weather_one_hot[weather] = 1
        
        # Construct state vector
        state_vector = np.concatenate((
            np.array([current_location / (self.env.m + 1)]),
            np.array([remaining_time]),
            visited,
            weather_one_hot
        ))
        return torch.FloatTensor(state_vector).to(self.device)
    
    def select_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
    
    def optimize_model(self, batch):
        """Optimize model using standard DQN update rule"""
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        
        batch_state_tensor = torch.stack([self.state_to_tensor(s) for s in batch_state])
        batch_action_tensor = torch.LongTensor(batch_action).to(self.device)
        batch_reward_tensor = torch.FloatTensor(batch_reward).to(self.device)
        batch_next_state_tensor = torch.stack([self.state_to_tensor(s) for s in batch_next_state])
        batch_done_tensor = torch.FloatTensor(batch_done).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(batch_state_tensor)
        state_action_values = q_values.gather(1, batch_action_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_state_tensor)
            next_state_values = next_q_values.max(1)[0]
        
        expected_state_action_values = batch_reward_tensor + (self.gamma * next_state_values * (1 - batch_done_tensor))
        
        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ShortestPathAgent:
    """
    Shortest path algorithm using Dijkstra's algorithm with time constraints
    """
    
    def __init__(self, env):
        self.env = env
        self.action_list = self._discretize_actions()
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}
        self.action_size = len(self.action_list)
    
    def _discretize_actions(self):
        """Discretize action space"""
        action_list = []
        for loc in range(self.env.m + 1):
            time_lower, time_upper = self.env.get_data_time_bounds(loc)
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=5)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
    
    def select_action(self, state, epsilon=0.0):
        """Select action using shortest path algorithm"""
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        visited = state['visited']
        
        # Find unvisited locations
        unvisited = []
        for loc in range(1, self.env.m + 1):
            if not visited[loc]:
                unvisited.append(loc)
        
        if not unvisited:
            # All locations visited, return home
            return self._find_action_to_location(0, 0)
        
        # Find shortest path to nearest unvisited location
        best_location = None
        best_cost = float('inf')
        
        for loc in unvisited:
            # Calculate cost to reach this location and return home
            travel_time = self.env.get_expected_flight_time(current_location, loc)
            data_time = self.env.get_data_time_bounds(loc)[0]  # Minimum data collection time
            return_time = self.env.get_expected_flight_time(loc, 0)
            
            total_time = travel_time + data_time + return_time
            
            # Check if feasible within remaining time
            if total_time <= remaining_time:
                # Use travel time as cost (could be enhanced with reward consideration)
                cost = travel_time
                if cost < best_cost:
                    best_cost = cost
                    best_location = loc
        
        if best_location is not None:
            # Find action to go to best location
            return self._find_action_to_location(best_location, self.env.get_data_time_bounds(best_location)[0])
        else:
            # No feasible location, return home
            return self._find_action_to_location(0, 0)
    
    def _find_action_to_location(self, target_location, data_time):
        """Find action index that corresponds to going to target location"""
        target_action = (target_location, data_time)
        
        # Find exact match
        for idx, action in self.action_index_mapping.items():
            if action == target_action:
                return idx
        
        # Find closest match if exact not found
        best_idx = None
        best_distance = float('inf')
        
        for idx, action in self.action_index_mapping.items():
            loc, time = action
            if loc == target_location:
                distance = abs(time - data_time)
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx
        
        return best_idx


class GeneticAlgorithmAgent:
    """
    Simple Genetic Algorithm for route planning
    """
    
    def __init__(self, env, population_size=20, generations=10):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.action_list = self._discretize_actions()
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}
        self.action_size = len(self.action_list)
    
    def _discretize_actions(self):
        """Discretize action space"""
        action_list = []
        for loc in range(self.env.m + 1):
            time_lower, time_upper = self.env.get_data_time_bounds(loc)
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=5)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
    
    def select_action(self, state, epsilon=0.0):
        """Select action using genetic algorithm planning"""
        current_location = state['current_location']
        visited = state['visited']
        remaining_time = state['remaining_time'][0]
        
        # Find unvisited locations
        unvisited = [loc for loc in range(1, self.env.m + 1) if not visited[loc]]
        
        if not unvisited:
            return self._find_action_to_location(0, 0)
        
        # Generate initial population of routes
        population = self._generate_initial_population(unvisited, current_location)
        
        # Evolve population
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_route(route, current_location, remaining_time) for route in population]
            
            # Selection and reproduction
            population = self._evolve_population(population, fitness_scores)
        
        # Select best route
        fitness_scores = [self._evaluate_route(route, current_location, remaining_time) for route in population]
        best_route = population[np.argmax(fitness_scores)]
        
        if best_route:
            next_location = best_route[0]
            data_time = self.env.get_data_time_bounds(next_location)[0]
            return self._find_action_to_location(next_location, data_time)
        else:
            return self._find_action_to_location(0, 0)
    
    def _generate_initial_population(self, unvisited, current_location):
        """Generate initial population of routes"""
        population = []
        for _ in range(self.population_size):
            route = random.sample(unvisited, min(len(unvisited), random.randint(1, len(unvisited))))
            population.append(route)
        return population
    
    def _evaluate_route(self, route, current_location, remaining_time):
        """Evaluate fitness of a route"""
        if not route:
            return 0
        
        total_time = 0
        total_reward = 0
        current_pos = current_location
        
        for location in route:
            # Travel time
            travel_time = self.env.get_expected_flight_time(current_pos, location)
            # Data collection time
            data_time = self.env.get_data_time_bounds(location)[0]
            
            total_time += travel_time + data_time
            
            # Simple reward calculation
            if self.env.criticality[location] == "HC":
                total_reward += 100
            else:
                total_reward += 50
            
            current_pos = location
        
        # Return time
        return_time = self.env.get_expected_flight_time(current_pos, 0)
        total_time += return_time
        
        # Penalty for exceeding time
        if total_time > remaining_time:
            return 0
        
        return total_reward / max(total_time, 1)  # Reward per unit time
    
    def _evolve_population(self, population, fitness_scores):
        """Evolve population using selection, crossover, and mutation"""
        new_population = []
        
        # Keep best individuals (elitism)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_size = self.population_size // 4
        
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]].copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Order crossover"""
        if not parent1 or not parent2:
            return random.choice([parent1, parent2]).copy()
        
        # Simple crossover: combine unique elements
        combined = list(set(parent1 + parent2))
        random.shuffle(combined)
        return combined[:random.randint(1, len(combined))]
    
    def _mutate(self, individual, mutation_rate=0.1):
        """Mutation operator"""
        if random.random() < mutation_rate and individual:
            if random.random() < 0.5 and len(individual) > 1:
                # Remove random element
                individual.pop(random.randint(0, len(individual) - 1))
            else:
                # Add random element (if possible)
                all_locations = list(range(1, self.env.m + 1))
                available = [loc for loc in all_locations if loc not in individual]
                if available:
                    individual.append(random.choice(available))
        
        return individual
    
    def _find_action_to_location(self, target_location, data_time):
        """Find action index that corresponds to going to target location"""
        target_action = (target_location, data_time)
        
        for idx, action in self.action_index_mapping.items():
            if action == target_action:
                return idx
        
        # Find closest match
        best_idx = None
        best_distance = float('inf')
        
        for idx, action in self.action_index_mapping.items():
            loc, time = action
            if loc == target_location:
                distance = abs(time - data_time)
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx
        
        return best_idx


class PriorityAgent:
    """
    Priority-based agent that selects locations based on criticality and proximity
    """
    
    def __init__(self, env):
        self.env = env
        self.action_list = self._discretize_actions()
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}
        self.action_size = len(self.action_list)
    
    def _discretize_actions(self):
        """Discretize action space"""
        action_list = []
        for loc in range(self.env.m + 1):
            time_lower, time_upper = self.env.get_data_time_bounds(loc)
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=5)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
    
    def select_action(self, state, epsilon=0.0):
        """Select action based on priority scoring"""
        current_location = state['current_location']
        visited = state['visited']
        remaining_time = state['remaining_time'][0]
        
        # Find unvisited locations
        unvisited = [loc for loc in range(1, self.env.m + 1) if not visited[loc]]
        
        if not unvisited:
            return self._find_action_to_location(0, 0)
        
        # Calculate priority scores for each unvisited location
        best_location = None
        best_score = -float('inf')
        
        for loc in unvisited:
            score = self._calculate_priority_score(current_location, loc, remaining_time)
            if score > best_score:
                best_score = score
                best_location = loc
        
        if best_location is not None:
            data_time = self.env.get_data_time_bounds(best_location)[0]
            return self._find_action_to_location(best_location, data_time)
        else:
            return self._find_action_to_location(0, 0)
    
    def _calculate_priority_score(self, current_location, target_location, remaining_time):
        """Calculate priority score for a location"""
        # Criticality bonus
        criticality_bonus = 2.0 if self.env.criticality[target_location] == "HC" else 1.0
        
        # Distance penalty (closer is better)
        travel_time = self.env.get_expected_flight_time(current_location, target_location)
        distance_penalty = 1.0 / (travel_time + 1)
        
        # Time feasibility check
        data_time = self.env.get_data_time_bounds(target_location)[0]
        return_time = self.env.get_expected_flight_time(target_location, 0)
        total_time_needed = travel_time + data_time + return_time
        
        if total_time_needed > remaining_time:
            return -float('inf')  # Not feasible
        
        # Time efficiency bonus (more time remaining is better)
        time_efficiency = remaining_time / (total_time_needed + 1)
        
        return criticality_bonus * distance_penalty * time_efficiency
    
    def _find_action_to_location(self, target_location, data_time):
        """Find action index that corresponds to going to target location"""
        target_action = (target_location, data_time)
        
        for idx, action in self.action_index_mapping.items():
            if action == target_action:
                return idx
        
        # Find closest match
        best_idx = None
        best_distance = float('inf')
        
        for idx, action in self.action_index_mapping.items():
            loc, time = action
            if loc == target_location:
                distance = abs(time - data_time)
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx
        
        return best_idx


# Network architectures
class DQN(nn.Module):
    """Standard DQN network"""
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


class DuelingDQN(nn.Module):
    """Dueling DQN network with separate value and advantage streams"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQN, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        # Value stream
        self.value_fc = nn.Linear(hidden_size, hidden_size)
        self.value_relu = nn.ReLU()
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, hidden_size)
        self.advantage_relu = nn.ReLU()
        self.advantage_head = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        # Shared layers
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        
        # Value stream
        value = self.value_relu(self.value_fc(x))
        value = self.value_head(value)
        
        # Advantage stream
        advantage = self.advantage_relu(self.advantage_fc(x))
        advantage = self.advantage_head(advantage)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values