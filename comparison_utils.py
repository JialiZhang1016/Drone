#!/usr/bin/env python3
"""
Utility Functions for Unified Comparison Framework

This module provides utility functions for:
- Configuration validation
- Result aggregation
- Performance metrics calculation
- Plotting utilities
- Statistical testing helpers
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

class ConfigValidator:
    """Validate comparison framework configurations"""
    
    @staticmethod
    def validate_config(config: Dict) -> Tuple[bool, List[str]]:
        """Validate configuration and return (is_valid, error_messages)"""
        errors = []
        
        # Check required sections
        required_sections = ['algorithms', 'environments', 'training', 'evaluation', 'output']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate algorithms
        if 'algorithms' in config:
            for alg_name, alg_config in config['algorithms'].items():
                if 'type' not in alg_config:
                    errors.append(f"Algorithm {alg_name} missing 'type' field")
                
                # Validate DQN components
                if alg_config.get('type') == 'dqn':
                    if 'components' not in alg_config:
                        errors.append(f"DQN algorithm {alg_name} missing 'components' field")
        
        # Validate environments
        if 'environments' in config:
            for env in config['environments']:
                if not isinstance(env, str) or not env.endswith('.json'):
                    errors.append(f"Invalid environment config: {env}")
        
        # Validate training parameters
        if 'training' in config:
            required_training = ['episodes', 'learning_rate', 'batch_size']
            for param in required_training:
                if param not in config['training']:
                    errors.append(f"Missing training parameter: {param}")
        
        # Validate evaluation parameters
        if 'evaluation' in config:
            if 'seeds' not in config['evaluation'] or not config['evaluation']['seeds']:
                errors.append("Missing or empty evaluation seeds")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def fix_common_issues(config: Dict) -> Dict:
        """Automatically fix common configuration issues"""
        fixed_config = config.copy()
        
        # Ensure default values for missing parameters
        if 'training' not in fixed_config:
            fixed_config['training'] = {}
        
        training_defaults = {
            'episodes': 2000,
            'batch_size': 64,
            'learning_rate': 0.001,
            'epsilon_start': 1.0,
            'epsilon_end': 0.02,
            'epsilon_decay': 0.995,
            'target_update_freq': 100,
            'memory_capacity': 10000,
            'gamma': 0.99
        }
        
        for key, default_value in training_defaults.items():
            if key not in fixed_config['training']:
                fixed_config['training'][key] = default_value
        
        # Ensure evaluation defaults
        if 'evaluation' not in fixed_config:
            fixed_config['evaluation'] = {}
        
        evaluation_defaults = {
            'episodes': 500,
            'final_eval_episodes': 100,
            'seeds': [42, 123, 456],
            'metrics': ["cumulative_reward", "success_rate", "episode_length", "training_time"]
        }
        
        for key, default_value in evaluation_defaults.items():
            if key not in fixed_config['evaluation']:
                fixed_config['evaluation'][key] = default_value
        
        return fixed_config


class ResultsAggregator:
    """Aggregate and process comparison results"""
    
    @staticmethod
    def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results by algorithm and environment"""
        
        aggregated = results_df.groupby(['algorithm', 'environment']).agg({
            'cumulative_reward': ['mean', 'std', 'min', 'max', 'count'],
            'success_rate': ['mean', 'std'],
            'episode_length': ['mean', 'std'],
            'training_time': ['mean', 'std'],
            'constraint_violations': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        
        return aggregated.reset_index()
    
    @staticmethod
    def calculate_rankings(results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate algorithm rankings across different metrics"""
        
        # Calculate mean performance for each algorithm
        algorithm_performance = results_df.groupby('algorithm').agg({
            'cumulative_reward': 'mean',
            'success_rate': 'mean',
            'episode_length': 'mean',
            'training_time': 'mean'
        }).round(3)
        
        # Calculate rankings (1 = best)
        rankings = pd.DataFrame()
        rankings['algorithm'] = algorithm_performance.index
        rankings['reward_rank'] = algorithm_performance['cumulative_reward'].rank(ascending=False)
        rankings['success_rank'] = algorithm_performance['success_rate'].rank(ascending=False)
        rankings['efficiency_rank'] = algorithm_performance['episode_length'].rank(ascending=True)  # Lower is better
        rankings['speed_rank'] = algorithm_performance['training_time'].rank(ascending=True)  # Lower is better
        
        # Calculate overall rank (simple average)
        rankings['overall_rank'] = (rankings['reward_rank'] + rankings['success_rank'] + 
                                   rankings['efficiency_rank'] + rankings['speed_rank']) / 4
        
        return rankings.sort_values('overall_rank')
    
    @staticmethod
    def identify_best_algorithms(results_df: pd.DataFrame, 
                                top_n: int = 3) -> Dict[str, List[str]]:
        """Identify best algorithms for different criteria"""
        
        algorithm_performance = results_df.groupby('algorithm').agg({
            'cumulative_reward': 'mean',
            'success_rate': 'mean',
            'episode_length': 'mean',
            'training_time': 'mean'
        })
        
        # Filter out non-trainable algorithms for training time comparison
        trainable_algorithms = algorithm_performance[algorithm_performance['training_time'] > 0]
        
        best_algorithms = {
            'highest_reward': algorithm_performance.nlargest(top_n, 'cumulative_reward').index.tolist(),
            'highest_success_rate': algorithm_performance.nlargest(top_n, 'success_rate').index.tolist(),
            'most_efficient': algorithm_performance.nsmallest(top_n, 'episode_length').index.tolist(),
            'fastest_training': trainable_algorithms.nsmallest(top_n, 'training_time').index.tolist() if not trainable_algorithms.empty else []
        }
        
        return best_algorithms


class PerformanceMetrics:
    """Calculate advanced performance metrics"""
    
    @staticmethod
    def calculate_efficiency_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various efficiency metrics"""
        
        efficiency_metrics = []
        
        for algorithm in results_df['algorithm'].unique():
            alg_data = results_df[results_df['algorithm'] == algorithm]
            
            avg_reward = alg_data['cumulative_reward'].mean()
            avg_success_rate = alg_data['success_rate'].mean()
            avg_episode_length = alg_data['episode_length'].mean()
            avg_training_time = alg_data['training_time'].mean()
            
            # Calculate efficiency metrics
            reward_per_step = avg_reward / max(avg_episode_length, 1)
            reward_per_second = avg_reward / max(avg_training_time, 1) if avg_training_time > 0 else np.inf
            success_per_step = avg_success_rate / max(avg_episode_length, 1)
            
            efficiency_metrics.append({
                'algorithm': algorithm,
                'reward_per_step': reward_per_step,
                'reward_per_second': reward_per_second,
                'success_per_step': success_per_step,
                'training_efficiency': reward_per_second if np.isfinite(reward_per_second) else 0
            })
        
        return pd.DataFrame(efficiency_metrics)
    
    @staticmethod
    def calculate_robustness_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate robustness metrics across environments"""
        
        robustness_metrics = []
        
        for algorithm in results_df['algorithm'].unique():
            alg_data = results_df[results_df['algorithm'] == algorithm]
            
            # Performance across environments
            env_performance = alg_data.groupby('environment')['cumulative_reward'].mean()
            env_success = alg_data.groupby('environment')['success_rate'].mean()
            
            # Robustness metrics
            performance_std = env_performance.std()
            performance_cv = performance_std / max(abs(env_performance.mean()), 1e-6)  # Coefficient of variation
            min_performance = env_performance.min()
            max_performance = env_performance.max()
            performance_range = max_performance - min_performance
            
            # Success rate robustness
            success_std = env_success.std()
            min_success = env_success.min()
            
            robustness_metrics.append({
                'algorithm': algorithm,
                'performance_std': performance_std,
                'performance_cv': performance_cv,
                'performance_range': performance_range,
                'min_performance': min_performance,
                'max_performance': max_performance,
                'success_std': success_std,
                'min_success_rate': min_success,
                'robustness_score': env_performance.mean() / max(performance_std, 1e-6)  # Higher is more robust
            })
        
        return pd.DataFrame(robustness_metrics)


class PlottingUtils:
    """Utility functions for creating standardized plots"""
    
    @staticmethod
    def get_algorithm_colors(algorithms: List[str]) -> Dict[str, str]:
        """Get consistent colors for algorithms across plots"""
        import matplotlib.pyplot as plt
        
        # Define color scheme
        color_map = {
            'dqn_vanilla': '#1f77b4',
            'dqn_reward': '#ff7f0e', 
            'dqn_mask': '#2ca02c',
            'dqn_full': '#d62728',
            'double_dqn': '#9467bd',
            'dueling_dqn': '#8c564b',
            'greedy': '#e377c2',
            'rule_based': '#7f7f7f',
            'shortest_path': '#bcbd22',
            'genetic_algorithm': '#17becf',
            'priority_based': '#ff9896',
            'random': '#c5b0d5'
        }
        
        # Get default color cycle for any missing algorithms
        default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        result_colors = {}
        color_index = 0
        
        for alg in algorithms:
            if alg in color_map:
                result_colors[alg] = color_map[alg]
            else:
                result_colors[alg] = default_colors[color_index % len(default_colors)]
                color_index += 1
        
        return result_colors
    
    @staticmethod
    def format_algorithm_name(algorithm_name: str) -> str:
        """Format algorithm names for display"""
        name_map = {
            'dqn_vanilla': 'Vanilla DQN',
            'dqn_reward': 'DQN + Reward Shaping',
            'dqn_mask': 'DQN + Action Masking',
            'dqn_full': 'Complete DQN',
            'double_dqn': 'Double DQN',
            'dueling_dqn': 'Dueling DQN',
            'greedy': 'Greedy Algorithm',
            'rule_based': 'Rule-based Algorithm',
            'shortest_path': 'Shortest Path',
            'genetic_algorithm': 'Genetic Algorithm',
            'priority_based': 'Priority-based',
            'random': 'Random Baseline'
        }
        
        return name_map.get(algorithm_name, algorithm_name.replace('_', ' ').title())


class StatisticalUtils:
    """Statistical testing utilities"""
    
    @staticmethod
    def perform_pairwise_tests(results_df: pd.DataFrame, 
                              metric: str = 'cumulative_reward',
                              test_type: str = 'wilcoxon') -> pd.DataFrame:
        """Perform pairwise statistical tests between algorithms"""
        from scipy.stats import wilcoxon, mannwhitneyu
        
        algorithms = results_df['algorithm'].unique()
        test_results = []
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:  # Avoid duplicate tests
                    data1 = results_df[results_df['algorithm'] == alg1][metric]
                    data2 = results_df[results_df['algorithm'] == alg2][metric]
                    
                    if len(data1) > 1 and len(data2) > 1:
                        try:
                            if test_type == 'wilcoxon':
                                stat, p_value = wilcoxon(data1, data2, alternative='two-sided')
                            else:  # mann-whitney
                                stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            # Effect size (Cohen's d)
                            effect_size = StatisticalUtils.cohens_d(data1, data2)
                            
                            test_results.append({
                                'algorithm_1': alg1,
                                'algorithm_2': alg2,
                                'statistic': stat,
                                'p_value': p_value,
                                'effect_size': effect_size,
                                'significant': p_value < 0.05,
                                'winner': alg1 if data1.mean() > data2.mean() else alg2,
                                'mean_difference': data1.mean() - data2.mean()
                            })
                        except Exception as e:
                            print(f"⚠️ Test failed for {alg1} vs {alg2}: {e}")
        
        return pd.DataFrame(test_results)
    
    @staticmethod
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1:
            return 0
        
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0
        
        return (group1.mean() - group2.mean()) / pooled_std
    
    @staticmethod
    def bonferroni_correction(p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple testing"""
        n_tests = len(p_values)
        return [min(p * n_tests, 1.0) for p in p_values]


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML or JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")


def save_config(config: Dict, output_path: str):
    """Save configuration to file"""
    with open(output_path, 'w') as f:
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif output_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported output file format: {output_path}")


def main():
    """Example usage of utility functions"""
    print("🔧 Comparison Framework Utilities")
    print("This module provides utility functions for the unified comparison framework.")
    print("Import specific classes and functions as needed.")


if __name__ == "__main__":
    main()