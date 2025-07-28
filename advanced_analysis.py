#!/usr/bin/env python3
"""
Advanced Analysis Tools for Unified Comparison Framework

This module provides comprehensive analysis and visualization tools:
- Statistical significance testing
- Algorithm ranking with confidence intervals
- Scalability analysis
- Robustness evaluation
- Performance heatmaps
- Convergence analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings("ignore")

class AdvancedAnalyzer:
    """
    Advanced analysis tools for algorithm comparison
    """
    
    def __init__(self, results_df: pd.DataFrame, output_dir: str, config: Dict = None):
        self.results_df = results_df
        self.output_dir = output_dir
        self.config = config or {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def run_comprehensive_analysis(self):
        """Run all analysis components"""
        print("🔬 Running Comprehensive Analysis...")
        
        # 1. Statistical Analysis
        self.perform_statistical_tests()
        
        # 2. Algorithm Ranking
        self.create_algorithm_ranking()
        
        # 3. Scalability Analysis
        self.analyze_scalability()
        
        # 4. Robustness Analysis
        self.analyze_robustness()
        
        # 5. Performance Visualization
        self.create_performance_visualizations()
        
        # 6. Convergence Analysis
        self.analyze_convergence()
        
        # 7. Computational Efficiency
        self.analyze_computational_efficiency()
        
        # 8. Generate Summary Report
        self.generate_comprehensive_report()
        
        print(f"✅ Analysis completed! Results saved in: {self.output_dir}")
    
    def perform_statistical_tests(self):
        """Perform comprehensive statistical significance testing"""
        print("📊 Performing Statistical Tests...")
        
        algorithms = self.results_df['algorithm'].unique()
        environments = self.results_df['environment'].unique()
        
        # Friedman test for each environment
        friedman_results = []
        
        for env in environments:
            env_data = self.results_df[self.results_df['environment'] == env]
            
            # Prepare data for Friedman test
            algorithm_groups = []
            for alg in algorithms:
                alg_data = env_data[env_data['algorithm'] == alg]['cumulative_reward']
                if len(alg_data) > 0:
                    algorithm_groups.append(alg_data.values)
            
            if len(algorithm_groups) > 2:
                try:
                    stat, p_value = friedmanchisquare(*algorithm_groups)
                    friedman_results.append({
                        'environment': env,
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'algorithms_tested': len(algorithm_groups)
                    })
                except Exception as e:
                    print(f"⚠️ Friedman test failed for {env}: {e}")
        
        # Pairwise comparisons (Wilcoxon signed-rank test)
        pairwise_results = []
        
        for env in environments:
            env_data = self.results_df[self.results_df['environment'] == env]
            
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms):
                    if i < j:  # Avoid duplicate comparisons
                        data1 = env_data[env_data['algorithm'] == alg1]['cumulative_reward']
                        data2 = env_data[env_data['algorithm'] == alg2]['cumulative_reward']
                        
                        if len(data1) > 1 and len(data2) > 1:
                            try:
                                # Wilcoxon signed-rank test
                                stat, p_value = wilcoxon(data1, data2, alternative='two-sided')
                                
                                # Effect size (Cohen's d)
                                effect_size = self.calculate_cohens_d(data1, data2)
                                
                                pairwise_results.append({
                                    'environment': env,
                                    'algorithm_1': alg1,
                                    'algorithm_2': alg2,
                                    'mean_1': data1.mean(),
                                    'mean_2': data2.mean(),
                                    'median_1': data1.median(),
                                    'median_2': data2.median(),
                                    'std_1': data1.std(),
                                    'std_2': data2.std(),
                                    'statistic': stat,
                                    'p_value': p_value,
                                    'effect_size': effect_size,
                                    'significant': p_value < 0.05,
                                    'winner': alg1 if data1.mean() > data2.mean() else alg2
                                })
                            except Exception as e:
                                print(f"⚠️ Pairwise test failed for {alg1} vs {alg2} in {env}: {e}")
        
        # Save results
        friedman_df = pd.DataFrame(friedman_results)
        pairwise_df = pd.DataFrame(pairwise_results)
        
        friedman_df.to_csv(os.path.join(self.output_dir, 'friedman_tests.csv'), index=False)
        pairwise_df.to_csv(os.path.join(self.output_dir, 'pairwise_comparisons.csv'), index=False)
        
        print(f"📈 Statistical tests completed")
        return friedman_df, pairwise_df
    
    def calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        return (group1.mean() - group2.mean()) / pooled_std
    
    def create_algorithm_ranking(self):
        """Create comprehensive algorithm ranking with confidence intervals"""
        print("🏆 Creating Algorithm Ranking...")
        
        # Calculate statistics for each algorithm
        ranking_data = []
        
        for algorithm in self.results_df['algorithm'].unique():
            alg_data = self.results_df[self.results_df['algorithm'] == algorithm]
            
            # Overall performance
            rewards = alg_data['cumulative_reward']
            success_rates = alg_data['success_rate']
            episode_lengths = alg_data['episode_length']
            training_times = alg_data['training_time']
            
            # Calculate confidence intervals
            reward_ci = stats.t.interval(0.95, len(rewards)-1, rewards.mean(), stats.sem(rewards))
            success_ci = stats.t.interval(0.95, len(success_rates)-1, success_rates.mean(), stats.sem(success_rates))
            
            ranking_data.append({
                'algorithm': algorithm,
                'mean_reward': rewards.mean(),
                'reward_std': rewards.std(),
                'reward_ci_lower': reward_ci[0],
                'reward_ci_upper': reward_ci[1],
                'mean_success_rate': success_rates.mean(),
                'success_std': success_rates.std(),
                'success_ci_lower': success_ci[0],
                'success_ci_upper': success_ci[1],
                'mean_episode_length': episode_lengths.mean(),
                'episode_length_std': episode_lengths.std(),
                'mean_training_time': training_times.mean(),
                'training_time_std': training_times.std(),
                'sample_size': len(alg_data)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('mean_reward', ascending=False)
        
        # Create ranking plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Performance Ranking with Confidence Intervals', fontsize=16)
        
        # Reward ranking
        ax1 = axes[0, 0]
        x_pos = np.arange(len(ranking_df))
        ax1.bar(x_pos, ranking_df['mean_reward'], 
                yerr=[ranking_df['mean_reward'] - ranking_df['reward_ci_lower'],
                      ranking_df['reward_ci_upper'] - ranking_df['mean_reward']],
                capsize=5, alpha=0.7)
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Mean Cumulative Reward')
        ax1.set_title('Reward Performance Ranking')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(ranking_df['algorithm'], rotation=45, ha='right')
        
        # Success rate ranking
        ax2 = axes[0, 1]
        ax2.bar(x_pos, ranking_df['mean_success_rate'],
                yerr=[ranking_df['mean_success_rate'] - ranking_df['success_ci_lower'],
                      ranking_df['success_ci_upper'] - ranking_df['mean_success_rate']],
                capsize=5, alpha=0.7, color='orange')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Mean Success Rate')
        ax2.set_title('Success Rate Ranking')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(ranking_df['algorithm'], rotation=45, ha='right')
        
        # Episode length
        ax3 = axes[1, 0]
        ax3.bar(x_pos, ranking_df['mean_episode_length'], alpha=0.7, color='green')
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Mean Episode Length')
        ax3.set_title('Efficiency Ranking (Lower is Better)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(ranking_df['algorithm'], rotation=45, ha='right')
        
        # Training time
        ax4 = axes[1, 1]
        ax4.bar(x_pos, ranking_df['mean_training_time'], alpha=0.7, color='red')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Mean Training Time (s)')
        ax4.set_title('Training Time Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(ranking_df['algorithm'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'algorithm_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save ranking data
        ranking_df.to_csv(os.path.join(self.output_dir, 'algorithm_ranking.csv'), index=False)
        
        return ranking_df
    
    def analyze_scalability(self):
        """Analyze algorithm scalability across environment complexity"""
        print("📈 Analyzing Scalability...")
        
        # Extract environment complexity (number of locations)
        self.results_df['num_locations'] = self.results_df['environment'].str.extract(r'config_(\d+)').astype(int)
        
        # Calculate mean performance for each algorithm-environment combination
        scalability_data = self.results_df.groupby(['algorithm', 'num_locations']).agg({
            'cumulative_reward': ['mean', 'std'],
            'success_rate': ['mean', 'std'],
            'episode_length': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        scalability_data.columns = ['_'.join(col).strip() for col in scalability_data.columns]
        scalability_data = scalability_data.reset_index()
        
        # Create scalability plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Scalability Analysis', fontsize=16)
        
        algorithms = self.results_df['algorithm'].unique()
        locations = sorted(self.results_df['num_locations'].unique())
        
        # Reward scalability
        ax1 = axes[0, 0]
        for alg in algorithms:
            alg_data = scalability_data[scalability_data['algorithm'] == alg]
            ax1.plot(alg_data['num_locations'], alg_data['cumulative_reward_mean'], 
                    marker='o', label=alg, linewidth=2)
            ax1.fill_between(alg_data['num_locations'],
                           alg_data['cumulative_reward_mean'] - alg_data['cumulative_reward_std'],
                           alg_data['cumulative_reward_mean'] + alg_data['cumulative_reward_std'],
                           alpha=0.2)
        ax1.set_xlabel('Number of Locations')
        ax1.set_ylabel('Mean Cumulative Reward')
        ax1.set_title('Reward Scalability')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Success rate scalability
        ax2 = axes[0, 1]
        for alg in algorithms:
            alg_data = scalability_data[scalability_data['algorithm'] == alg]
            ax2.plot(alg_data['num_locations'], alg_data['success_rate_mean'],
                    marker='s', label=alg, linewidth=2)
        ax2.set_xlabel('Number of Locations')
        ax2.set_ylabel('Mean Success Rate')
        ax2.set_title('Success Rate Scalability')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Episode length scalability
        ax3 = axes[1, 0]
        for alg in algorithms:
            alg_data = scalability_data[scalability_data['algorithm'] == alg]
            ax3.plot(alg_data['num_locations'], alg_data['episode_length_mean'],
                    marker='^', label=alg, linewidth=2)
        ax3.set_xlabel('Number of Locations')
        ax3.set_ylabel('Mean Episode Length')
        ax3.set_title('Episode Length vs Complexity')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Training time scalability
        ax4 = axes[1, 1]
        for alg in algorithms:
            alg_data = scalability_data[scalability_data['algorithm'] == alg]
            if alg_data['training_time_mean'].max() > 0:  # Only plot algorithms that require training
                ax4.plot(alg_data['num_locations'], alg_data['training_time_mean'],
                        marker='d', label=alg, linewidth=2)
        ax4.set_xlabel('Number of Locations')
        ax4.set_ylabel('Mean Training Time (s)')
        ax4.set_title('Training Time Scalability')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save scalability data
        scalability_data.to_csv(os.path.join(self.output_dir, 'scalability_data.csv'), index=False)
        
        return scalability_data
    
    def analyze_robustness(self):
        """Analyze algorithm robustness across different conditions"""
        print("🛡️ Analyzing Robustness...")
        
        # Create robustness heatmap
        pivot_data = self.results_df.pivot_table(
            values='cumulative_reward',
            index='algorithm',
            columns='environment',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax)
        ax.set_title('Algorithm Performance Heatmap Across Environments')
        ax.set_xlabel('Environment Configuration')
        ax.set_ylabel('Algorithm')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'robustness_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate robustness metrics
        robustness_metrics = []
        
        for algorithm in self.results_df['algorithm'].unique():
            alg_data = self.results_df[self.results_df['algorithm'] == algorithm]
            
            # Performance variance across environments
            env_performance = alg_data.groupby('environment')['cumulative_reward'].mean()
            
            robustness_metrics.append({
                'algorithm': algorithm,
                'mean_performance': env_performance.mean(),
                'std_across_envs': env_performance.std(),
                'min_performance': env_performance.min(),
                'max_performance': env_performance.max(),
                'robustness_score': env_performance.mean() / (env_performance.std() + 1e-6)  # Higher is more robust
            })
        
        robustness_df = pd.DataFrame(robustness_metrics)
        robustness_df = robustness_df.sort_values('robustness_score', ascending=False)
        
        # Save robustness data
        robustness_df.to_csv(os.path.join(self.output_dir, 'robustness_analysis.csv'), index=False)
        
        return robustness_df
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations"""
        print("🎨 Creating Performance Visualizations...")
        
        # Box plots for reward distribution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Distribution Analysis', fontsize=16)
        
        # Reward distribution
        ax1 = axes[0, 0]
        self.results_df.boxplot(column='cumulative_reward', by='algorithm', ax=ax1)
        ax1.set_title('Reward Distribution by Algorithm')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Cumulative Reward')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Success rate distribution
        ax2 = axes[0, 1]
        self.results_df.boxplot(column='success_rate', by='algorithm', ax=ax2)
        ax2.set_title('Success Rate Distribution by Algorithm')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Success Rate')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Episode length distribution
        ax3 = axes[1, 0]
        self.results_df.boxplot(column='episode_length', by='algorithm', ax=ax3)
        ax3.set_title('Episode Length Distribution by Algorithm')
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Episode Length')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Training time distribution
        ax4 = axes[1, 1]
        training_data = self.results_df[self.results_df['training_time'] > 0]
        if not training_data.empty:
            training_data.boxplot(column='training_time', by='algorithm', ax=ax4)
            ax4.set_title('Training Time Distribution by Algorithm')
            ax4.set_xlabel('Algorithm')
            ax4.set_ylabel('Training Time (s)')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_convergence(self):
        """Analyze convergence patterns for RL algorithms"""
        print("📊 Analyzing Convergence Patterns...")
        
        # This would require episode-by-episode data
        # For now, create a placeholder analysis
        print("⚠️ Convergence analysis requires episode-by-episode data")
        print("💡 Consider modifying training loop to save convergence data")
    
    def analyze_computational_efficiency(self):
        """Analyze computational efficiency metrics"""
        print("⚡ Analyzing Computational Efficiency...")
        
        efficiency_data = []
        
        for algorithm in self.results_df['algorithm'].unique():
            alg_data = self.results_df[self.results_df['algorithm'] == algorithm]
            
            avg_reward = alg_data['cumulative_reward'].mean()
            avg_training_time = alg_data['training_time'].mean()
            
            # Efficiency metric: reward per second of training
            efficiency = avg_reward / (avg_training_time + 1e-6)
            
            efficiency_data.append({
                'algorithm': algorithm,
                'avg_reward': avg_reward,
                'avg_training_time': avg_training_time,
                'efficiency_score': efficiency,
                'is_trainable': avg_training_time > 0
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df = efficiency_df.sort_values('efficiency_score', ascending=False)
        
        # Create efficiency plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Efficiency score
        trainable_algs = efficiency_df[efficiency_df['is_trainable']]
        if not trainable_algs.empty:
            ax1.bar(range(len(trainable_algs)), trainable_algs['efficiency_score'])
            ax1.set_xlabel('Algorithm')
            ax1.set_ylabel('Efficiency (Reward/Second)')
            ax1.set_title('Training Efficiency Comparison')
            ax1.set_xticks(range(len(trainable_algs)))
            ax1.set_xticklabels(trainable_algs['algorithm'], rotation=45, ha='right')
        
        # Scatter plot: reward vs training time
        ax2.scatter(efficiency_df['avg_training_time'], efficiency_df['avg_reward'], s=100, alpha=0.7)
        for i, alg in enumerate(efficiency_df['algorithm']):
            ax2.annotate(alg, (efficiency_df.iloc[i]['avg_training_time'], 
                              efficiency_df.iloc[i]['avg_reward']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Average Training Time (s)')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Reward vs Training Time Trade-off')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'computational_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save efficiency data
        efficiency_df.to_csv(os.path.join(self.output_dir, 'computational_efficiency.csv'), index=False)
        
        return efficiency_df
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("📝 Generating Comprehensive Report...")
        
        report_lines = []
        report_lines.append("# Comprehensive Algorithm Comparison Report\n")
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        report_lines.append("## Executive Summary\n")
        report_lines.append(f"- Total Experiments: {len(self.results_df)}\n")
        report_lines.append(f"- Algorithms Tested: {len(self.results_df['algorithm'].unique())}\n")
        report_lines.append(f"- Environments: {len(self.results_df['environment'].unique())}\n")
        report_lines.append(f"- Random Seeds: {len(self.results_df['seed'].unique())}\n\n")
        
        # Top Performers
        report_lines.append("## Top Performing Algorithms\n")
        top_performers = self.results_df.groupby('algorithm')['cumulative_reward'].mean().sort_values(ascending=False)
        
        for i, (alg, score) in enumerate(top_performers.head(5).items(), 1):
            report_lines.append(f"{i}. **{alg}**: {score:.2f} average reward\n")
        
        report_lines.append("\n")
        
        # Statistical Significance
        report_lines.append("## Statistical Significance\n")
        report_lines.append("Detailed statistical tests are available in separate CSV files:\n")
        report_lines.append("- `friedman_tests.csv`: Friedman tests for each environment\n")
        report_lines.append("- `pairwise_comparisons.csv`: Pairwise algorithm comparisons\n")
        report_lines.append("- `algorithm_ranking.csv`: Complete ranking with confidence intervals\n\n")
        
        # Scalability Insights
        report_lines.append("## Scalability Analysis\n")
        if 'num_locations' in self.results_df.columns:
            scalability_summary = self.results_df.groupby(['algorithm', 'num_locations'])['cumulative_reward'].mean().unstack()
            
            report_lines.append("Performance scaling with environment complexity:\n")
            for alg in scalability_summary.index:
                perf_change = scalability_summary.loc[alg].iloc[-1] - scalability_summary.loc[alg].iloc[0]
                trend = "↗️" if perf_change > 0 else "↘️" if perf_change < 0 else "→"
                report_lines.append(f"- **{alg}**: {trend} ({perf_change:+.2f} change)\n")
        
        report_lines.append("\n")
        
        # Recommendations
        report_lines.append("## Recommendations\n")
        report_lines.append("Based on the comprehensive analysis:\n\n")
        
        # Best overall
        best_overall = top_performers.index[0]
        report_lines.append(f"1. **Best Overall Performance**: {best_overall}\n")
        
        # Most efficient
        training_algorithms = self.results_df[self.results_df['training_time'] > 0]
        if not training_algorithms.empty:
            efficiency = training_algorithms.groupby('algorithm').apply(
                lambda x: x['cumulative_reward'].mean() / x['training_time'].mean()
            ).sort_values(ascending=False)
            most_efficient = efficiency.index[0]
            report_lines.append(f"2. **Most Training Efficient**: {most_efficient}\n")
        
        # Most robust
        robustness = self.results_df.groupby('algorithm')['cumulative_reward'].std().sort_values()
        most_robust = robustness.index[0]
        report_lines.append(f"3. **Most Robust**: {most_robust} (lowest variance)\n")
        
        report_lines.append("\n")
        report_lines.append("## Files Generated\n")
        report_lines.append("- `algorithm_ranking.png`: Performance ranking visualization\n")
        report_lines.append("- `scalability_analysis.png`: Scalability across environments\n")
        report_lines.append("- `robustness_heatmap.png`: Performance heatmap\n")
        report_lines.append("- `performance_distributions.png`: Statistical distributions\n")
        report_lines.append("- `computational_efficiency.png`: Training efficiency analysis\n")
        report_lines.append("- Various CSV files with detailed numerical results\n")
        
        # Save report
        with open(os.path.join(self.output_dir, 'comprehensive_report.md'), 'w') as f:
            f.writelines(report_lines)
        
        print("📋 Comprehensive report generated")


def main():
    """Example usage of advanced analysis tools"""
    # This would be called from the unified comparison framework
    pass

if __name__ == "__main__":
    main()