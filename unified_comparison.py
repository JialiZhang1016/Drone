#!/usr/bin/env python3
"""
Unified Comparison Framework for Drone Route Planning Algorithms

This framework provides a standardized way to compare different algorithms:
- Deep RL: DQN variants, PPO, A2C, SAC
- Heuristic: Greedy, Rule-based
- Traditional: Random baseline, shortest path
- New baselines: DDQN, Dueling DQN

Features:
- Unified training conditions
- Standardized evaluation metrics  
- Fair comparison mechanisms
- Statistical significance testing
- Reproducible results
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import torch
import warnings
warnings.filterwarnings("ignore")

# Import environment and existing agents
from drone_env import PureDroneRoutePlanningEnv
from agent.dqn_agent_ablation import IntelligentDQNAgent
from agent.heuristic_agents import GreedyAgent, RuleBasedAgent

class UnifiedComparisonFramework:
    """
    Unified framework for comparing all algorithms under fair conditions
    """
    
    def __init__(self, config_path: str = "comparison_config.yaml"):
        """Initialize the comparison framework"""
        self.config = self.load_config(config_path)
        self.results = defaultdict(list)
        self.algorithms = {}
        self.environments = {}
        
        # Set up reproducibility
        self.setup_reproducibility()
        
        print("🚀 Unified Comparison Framework Initialized")
        print(f"📊 Algorithms to compare: {list(self.config['algorithms'].keys())}")
        print(f"🌍 Environment configs: {self.config['environments']}")
        print(f"🎲 Random seeds: {self.config['evaluation']['seeds']}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load comparison configuration"""
        # Default configuration if file doesn't exist
        default_config = {
            "algorithms": {
                "dqn_vanilla": {"type": "dqn", "components": {"action_mask": False, "safety": False, "constraint": False, "reward_shaping": False}},
                "dqn_reward": {"type": "dqn", "components": {"action_mask": False, "safety": False, "constraint": False, "reward_shaping": True}},
                "dqn_mask": {"type": "dqn", "components": {"action_mask": True, "safety": False, "constraint": False, "reward_shaping": False}},
                "dqn_full": {"type": "dqn", "components": {"action_mask": True, "safety": True, "constraint": True, "reward_shaping": True}},
                "greedy": {"type": "heuristic", "class": "GreedyAgent"},
                "rule_based": {"type": "heuristic", "class": "RuleBasedAgent"},
                "random": {"type": "baseline", "class": "RandomAgent"}
            },
            "environments": ["config_5.json", "config_8.json", "config_10.json"],
            "training": {
                "episodes": 2000,
                "batch_size": 64,
                "learning_rate": 0.001,
                "epsilon_start": 1.0,
                "epsilon_end": 0.02,
                "epsilon_decay": 0.995,
                "target_update_freq": 100,
                "memory_capacity": 10000
            },
            "evaluation": {
                "episodes": 500,
                "seeds": [42, 123, 456],
                "final_eval_episodes": 100,
                "metrics": ["cumulative_reward", "success_rate", "episode_length", "training_time", "constraint_violations"]
            },
            "output": {
                "results_dir": "unified_comparison_results",
                "save_models": True,
                "generate_plots": True,
                "statistical_tests": True
            }
        }
        
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(loaded_config)
            except ImportError:
                print("⚠️ PyYAML not available, using default configuration")
            except Exception as e:
                print(f"⚠️ Error loading config: {e}, using default configuration")
        
        return default_config
    
    def setup_reproducibility(self):
        """Set up reproducible random seeds"""
        base_seed = self.config['evaluation']['seeds'][0]
        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(base_seed)
            torch.cuda.manual_seed_all(base_seed)
        
        print(f"🎲 Reproducibility set up with base seed: {base_seed}")
    
    def create_environment(self, config_file: str) -> PureDroneRoutePlanningEnv:
        """Create standardized environment"""
        config_path = os.path.join("config", config_file)
        
        if not os.path.exists(config_path):
            print(f"⚠️ Config file {config_path} not found, creating default")
            self.create_default_config(config_path)
        
        with open(config_path, 'r') as f:
            env_config = json.load(f)
        
        return PureDroneRoutePlanningEnv(env_config, use_reward_shaping=False)
    
    def create_default_config(self, config_path: str):
        """Create a default configuration file"""
        # Extract number of locations from filename
        import re
        match = re.search(r'config_(\d+)\.json', config_path)
        num_locations = int(match.group(1)) if match else 5
        
        default_config = {
            "num_locations": num_locations,
            "T_max": 2500 + (num_locations - 5) * 400,
            "weather_prob": 0.8,
            "extreme_weather_prob": 0.05,
            "P_penalty": int(round(8000 * (num_locations / 8.0)**1.5 / 1000) * 1000),
            "T_flight_good": [[0.0 for _ in range(num_locations + 1)] for _ in range(num_locations + 1)],
            "bad_weather_delay_factor": [1.1, 1.5],
            "extreme_weather_delay_factor": [1.8, 2.5],
            "T_data_lower": [0] + [15 for _ in range(num_locations)],
            "T_data_upper": [0] + [25 for _ in range(num_locations)],
            "criticality": ["HC"] + ["HC" for _ in range(num_locations)]
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    def create_agent(self, algorithm_name: str, algorithm_config: Dict, env: PureDroneRoutePlanningEnv):
        """Create agent based on algorithm configuration"""
        algo_type = algorithm_config['type']
        
        if algo_type == 'dqn':
            components = algorithm_config['components']
            return IntelligentDQNAgent(
                env=env,
                num_time_bins=5,
                hidden_size=128,
                learning_rate=self.config['training']['learning_rate'],
                gamma=self.config['training']['gamma'],
                use_action_mask=components.get('action_mask', False),
                use_safety_mechanism=components.get('safety', False),
                use_constraint_check=components.get('constraint', False)
            )
        
        elif algo_type == 'advanced_rl':
            agent_class = algorithm_config['class']
            if agent_class == 'DoubleDQNAgent':
                from agent.advanced_agents import DoubleDQNAgent
                return DoubleDQNAgent(
                    env=env,
                    num_time_bins=5,
                    hidden_size=128,
                    learning_rate=self.config['training']['learning_rate'],
                    gamma=self.config['training']['gamma']
                )
            elif agent_class == 'DuelingDQNAgent':
                from agent.advanced_agents import DuelingDQNAgent
                return DuelingDQNAgent(
                    env=env,
                    num_time_bins=5,
                    hidden_size=128,
                    learning_rate=self.config['training']['learning_rate'],
                    gamma=self.config['training']['gamma']
                )
        
        elif algo_type == 'heuristic':
            agent_class = algorithm_config['class']
            # Create action list for heuristic agents
            action_list = self._create_action_list(env)
            if agent_class == 'GreedyAgent':
                return GreedyAgent(env, action_list)
            elif agent_class == 'RuleBasedAgent':
                return RuleBasedAgent(env, action_list)
        
        elif algo_type == 'path_planning':
            agent_class = algorithm_config['class']
            if agent_class == 'ShortestPathAgent':
                from agent.advanced_agents import ShortestPathAgent
                return ShortestPathAgent(env)
        
        elif algo_type == 'optimization':
            agent_class = algorithm_config['class']
            if agent_class == 'GeneticAlgorithmAgent':
                from agent.advanced_agents import GeneticAlgorithmAgent
                return GeneticAlgorithmAgent(env)
        
        elif algo_type == 'priority':
            agent_class = algorithm_config['class']
            if agent_class == 'PriorityAgent':
                from agent.advanced_agents import PriorityAgent
                return PriorityAgent(env)
        
        elif algo_type == 'baseline':
            agent_class = algorithm_config['class']
            if agent_class == 'RandomAgent':
                return RandomAgent(env)
        
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}")
    
    def _create_action_list(self, env):
        """Create discretized action list for agents"""
        action_list = []
        num_time_bins = 5
        
        for loc in range(env.m + 1):  # Including Home
            # Get data collection time bounds
            time_lower = env.T_data_lower[loc]
            time_upper = env.T_data_upper[loc]
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=num_time_bins)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        
        return action_list
    
    def train_agent(self, agent, env: PureDroneRoutePlanningEnv, algorithm_name: str, seed: int) -> Dict:
        """Train an agent and return training metrics"""
        print(f"🔄 Training {algorithm_name} with seed {seed}...")
        
        start_time = time.time()
        
        # Skip training for heuristic and baseline agents
        if hasattr(agent, 'select_action') and not hasattr(agent, 'optimize_model'):
            print(f"⏩ Skipping training for heuristic/baseline agent: {algorithm_name}")
            return {"training_time": 0.0, "training_episodes": 0}
        
        # Training loop for RL agents
        training_config = self.config['training']
        num_episodes = training_config['episodes']
        
        epsilon = training_config['epsilon_start']
        epsilon_decay = training_config['epsilon_decay']
        epsilon_min = training_config['epsilon_end']
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, _ = env.reset(seed=seed + episode)
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action_idx = agent.select_action(state, epsilon)
                
                if action_idx is None:
                    break
                
                action = agent.action_index_mapping[action_idx]
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Store experience for RL agents
                if hasattr(agent, 'memory'):
                    agent.memory.append((state, action_idx, reward, next_state, done))
                
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Update epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Train the agent
            if hasattr(agent, 'optimize_model') and len(getattr(agent, 'memory', [])) >= training_config['batch_size']:
                batch = random.sample(agent.memory, training_config['batch_size'])
                agent.optimize_model(batch)
            
            # Update target network
            if hasattr(agent, 'target_net') and episode % training_config['target_update_freq'] == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            # Progress logging
            if episode % 200 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                print(f"  Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        return {
            "training_time": training_time,
            "training_episodes": num_episodes,
            "final_avg_reward": np.mean(episode_rewards[-100:]) if episode_rewards else 0,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }
    
    def evaluate_agent(self, agent, env: PureDroneRoutePlanningEnv, algorithm_name: str, seed: int) -> Dict:
        """Evaluate an agent and return evaluation metrics"""
        print(f"📊 Evaluating {algorithm_name} with seed {seed}...")
        
        eval_episodes = self.config['evaluation']['final_eval_episodes']
        
        rewards = []
        success_rates = []
        episode_lengths = []
        constraint_violations = []
        
        for episode in range(eval_episodes):
            state, _ = env.reset(seed=seed + 10000 + episode)  # Different seed for evaluation
            episode_reward = 0
            episode_length = 0
            violations = 0
            done = False
            
            while not done:
                # Use greedy policy (epsilon=0) for evaluation
                if hasattr(agent, 'select_action'):
                    # Check if agent's select_action accepts epsilon parameter
                    import inspect
                    sig = inspect.signature(agent.select_action)
                    if 'epsilon' in sig.parameters:
                        action_idx = agent.select_action(state, epsilon=0.0)
                    else:
                        action_idx = agent.select_action(state)
                else:
                    action_idx = None
                
                if action_idx is None:
                    break
                
                action = agent.action_index_mapping[action_idx]
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Count constraint violations
                if 'constraint_violation' in info and info['constraint_violation']:
                    violations += 1
                
                state = next_state
            
            rewards.append(episode_reward)
            success_rates.append(1.0 if env.L_t == 0 else 0.0)  # Successfully returned home
            episode_lengths.append(episode_length)
            constraint_violations.append(violations)
        
        return {
            "cumulative_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "success_rate": np.mean(success_rates),
            "episode_length": np.mean(episode_lengths),
            "episode_length_std": np.std(episode_lengths),
            "constraint_violations": np.mean(constraint_violations),
            "all_rewards": rewards,
            "all_success_rates": success_rates,
            "all_episode_lengths": episode_lengths
        }
    
    def run_single_comparison(self, env_config: str, algorithm_name: str, seed: int) -> Dict:
        """Run a single algorithm on a single environment with a single seed"""
        print(f"\n{'='*60}")
        print(f"🔬 Running: {algorithm_name} on {env_config} (seed={seed})")
        print(f"{'='*60}")
        
        # Create environment
        env = self.create_environment(env_config)
        
        # Create agent
        algorithm_config = self.config['algorithms'][algorithm_name]
        agent = self.create_agent(algorithm_name, algorithm_config, env)
        
        # Train agent
        training_results = self.train_agent(agent, env, algorithm_name, seed)
        
        # Evaluate agent  
        evaluation_results = self.evaluate_agent(agent, env, algorithm_name, seed)
        
        # Combine results
        results = {
            "algorithm": algorithm_name,
            "environment": env_config,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            **training_results,
            **evaluation_results
        }
        
        return results
    
    def run_full_comparison(self) -> pd.DataFrame:
        """Run complete comparison across all algorithms, environments, and seeds"""
        print("\n🚀 Starting Full Comparison Study")
        print(f"Total experiments: {len(self.config['algorithms']) * len(self.config['environments']) * len(self.config['evaluation']['seeds'])}")
        
        all_results = []
        total_experiments = 0
        
        for env_config in self.config['environments']:
            for algorithm_name in self.config['algorithms'].keys():
                for seed in self.config['evaluation']['seeds']:
                    try:
                        result = self.run_single_comparison(env_config, algorithm_name, seed)
                        all_results.append(result)
                        total_experiments += 1
                        
                        print(f"✅ Completed {total_experiments} experiments")
                        
                    except Exception as e:
                        print(f"❌ Error in {algorithm_name} on {env_config} (seed={seed}): {e}")
                        continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        self.save_results(results_df)
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame):
        """Save results to files"""
        output_dir = self.config['output']['results_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(output_dir, "comparison_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"💾 Results saved to: {csv_path}")
        
        # Save configuration
        config_path = os.path.join(output_dir, "comparison_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return output_dir
    
    def generate_analysis(self, results_df: pd.DataFrame, output_dir: str):
        """Generate comprehensive analysis and visualization"""
        print("\n📊 Generating Analysis and Visualizations...")
        
        # Basic analysis
        self.plot_performance_comparison(results_df, output_dir)
        self.perform_statistical_analysis(results_df, output_dir)
        self.generate_summary_report(results_df, output_dir)
        
        # Advanced analysis using AdvancedAnalyzer
        try:
            from advanced_analysis import AdvancedAnalyzer
            advanced_analyzer = AdvancedAnalyzer(results_df, output_dir, self.config)
            advanced_analyzer.run_comprehensive_analysis()
        except ImportError as e:
            print(f"⚠️ Advanced analysis not available: {e}")
        except Exception as e:
            print(f"⚠️ Error in advanced analysis: {e}")
    
    def plot_performance_comparison(self, results_df: pd.DataFrame, output_dir: str):
        """Create performance comparison plots"""
        # Group by algorithm and environment
        grouped = results_df.groupby(['algorithm', 'environment']).agg({
            'cumulative_reward': ['mean', 'std'],
            'success_rate': ['mean', 'std'],
            'episode_length': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
        grouped = grouped.reset_index()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16)
        
        metrics = ['cumulative_reward', 'success_rate', 'episode_length', 'training_time']
        titles = ['Cumulative Reward', 'Success Rate', 'Episode Length', 'Training Time (s)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Create bar plot with error bars
            pivot_data = grouped.pivot(index='environment', columns='algorithm', values=f'{metric}_mean')
            pivot_std = grouped.pivot(index='environment', columns='algorithm', values=f'{metric}_std')
            
            pivot_data.plot(kind='bar', ax=ax, yerr=pivot_std, capsize=4)
            ax.set_title(title)
            ax.set_xlabel('Environment')
            ax.set_ylabel(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Performance comparison plot saved")
    
    def perform_statistical_analysis(self, results_df: pd.DataFrame, output_dir: str):
        """Perform statistical significance testing"""
        from scipy import stats
        
        algorithms = results_df['algorithm'].unique()
        environments = results_df['environment'].unique()
        
        statistical_results = []
        
        for env in environments:
            env_data = results_df[results_df['environment'] == env]
            
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms):
                    if i < j:  # Avoid duplicate comparisons
                        data1 = env_data[env_data['algorithm'] == alg1]['cumulative_reward']
                        data2 = env_data[env_data['algorithm'] == alg2]['cumulative_reward']
                        
                        if len(data1) > 1 and len(data2) > 1:
                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(data1, data2)
                            
                            statistical_results.append({
                                'environment': env,
                                'algorithm_1': alg1,
                                'algorithm_2': alg2,
                                'mean_1': data1.mean(),
                                'mean_2': data2.mean(),
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
        
        # Save statistical results
        stats_df = pd.DataFrame(statistical_results)
        stats_path = os.path.join(output_dir, 'statistical_analysis.csv')
        stats_df.to_csv(stats_path, index=False)
        print("📊 Statistical analysis saved")
    
    def generate_summary_report(self, results_df: pd.DataFrame, output_dir: str):
        """Generate a summary report"""
        report = []
        report.append("# Unified Comparison Framework - Summary Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Total Experiments: {len(results_df)}\n")
        
        # Algorithm ranking
        report.append("\n## Algorithm Performance Ranking\n")
        
        ranking = results_df.groupby('algorithm').agg({
            'cumulative_reward': ['mean', 'std'],
            'success_rate': ['mean', 'std'],
            'training_time': 'mean'
        }).round(3)
        
        ranking.columns = ['reward_mean', 'reward_std', 'success_mean', 'success_std', 'time_mean']
        ranking = ranking.sort_values('reward_mean', ascending=False)
        
        for i, (alg, row) in enumerate(ranking.iterrows(), 1):
            report.append(f"{i}. **{alg}**: Reward={row['reward_mean']:.2f}±{row['reward_std']:.2f}, "
                         f"Success={row['success_mean']:.2f}±{row['success_std']:.2f}, "
                         f"Time={row['time_mean']:.1f}s\n")
        
        # Environment complexity analysis
        report.append("\n## Environment Complexity Analysis\n")
        env_analysis = results_df.groupby('environment')['cumulative_reward'].mean().sort_values(ascending=False)
        
        for env, reward in env_analysis.items():
            report.append(f"- **{env}**: Average Reward = {reward:.2f}\n")
        
        # Save report
        report_path = os.path.join(output_dir, 'summary_report.md')
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        print("📝 Summary report generated")


class RandomAgent:
    """Random baseline agent for comparison"""
    
    def __init__(self, env):
        self.env = env
        self.action_list = self._discretize_actions()
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}
        self.action_size = len(self.action_list)
    
    def _discretize_actions(self):
        """Discretize action space - same as DQN agent"""
        action_list = []
        for loc in range(self.env.m + 1):
            time_lower = self.env.T_data_lower[loc]
            time_upper = self.env.T_data_upper[loc]
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=5)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
    
    def select_action(self, state, epsilon=0.0):
        """Select random valid action"""
        valid_actions = []
        
        for idx, action in self.action_index_mapping.items():
            loc, t_data = action
            
            # Basic validity check - not visited and within bounds
            if loc != 0 and self.env.is_location_visited(loc):
                continue
            
            time_lower = self.env.T_data_lower[loc]
            time_upper = self.env.T_data_upper[loc]
            if not (time_lower <= t_data <= time_upper):
                continue
            
            valid_actions.append(idx)
        
        if not valid_actions:
            return None
        
        return random.choice(valid_actions)


def main():
    """Main execution function"""
    print("🚀 Starting Unified Comparison Framework")
    
    # Initialize framework
    framework = UnifiedComparisonFramework()
    
    # Run complete comparison
    results_df = framework.run_full_comparison()
    
    # Generate analysis
    output_dir = framework.save_results(results_df)
    framework.generate_analysis(results_df, output_dir)
    
    print(f"\n🎉 Comparison completed! Results saved in: {output_dir}")
    print("\n📊 Quick Summary:")
    summary = results_df.groupby('algorithm')['cumulative_reward'].mean().sort_values(ascending=False)
    for i, (alg, reward) in enumerate(summary.items(), 1):
        print(f"  {i}. {alg}: {reward:.2f}")


if __name__ == "__main__":
    main()