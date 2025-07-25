# ablation_experiment.py

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from collections import deque
import random
import torch
import os

from ablation_drone_env import AblationDroneRoutePlanningEnv
from ablation_dqn_agent import AblationDQNAgent

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class ExperienceReplay:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train_ablation_model(model_config, num_episodes=1000, seed=42):
    """Train a single ablation model configuration"""
    set_random_seed(seed)
    
    # Load environment configuration
    with open('../current_version/config/realword_8/config_8.json', 'r') as f:
        env_config = json.load(f)
    
    # Create environment with ablation settings
    env = AblationDroneRoutePlanningEnv(
        env_config, 
        use_safety_mechanism=model_config['use_safety_mechanism'],
        use_constraint_check=model_config['use_constraint_check']
    )
    
    # Create agent with ablation settings
    agent = AblationDQNAgent(
        env,
        use_action_mask=model_config['use_action_mask'],
        use_safety_mechanism=model_config['use_safety_mechanism'],
        use_constraint_check=model_config['use_constraint_check']
    )
    
    # Training parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 100
    batch_size = 32
    
    # Experience replay
    memory = ExperienceReplay(capacity=10000)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    invalid_actions_rate = []
    safety_violations_rate = []
    constraint_violations_rate = []
    
    epsilon = epsilon_start
    
    print(f"Training {model_config['name']}...")
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            action_idx = agent.select_action(state, epsilon)
            
            if action_idx is None:
                # No valid actions available
                break
            
            action = agent.action_index_mapping[action_idx]
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            memory.push(state, action_idx, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            episode_length += 1
            
            # Train the agent
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                agent.optimize_model(batch)
            
            if done:
                break
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        # Calculate success rate (returned to home safely)
        current_location = info.get('next_state', {}).get('current_location', state['current_location'])
        success = (current_location == 0)
        success_rate.append(1 if success else 0)
        
        # Record violation rates
        invalid_actions_rate.append(info.get('invalid_actions', 0))
        safety_violations_rate.append(info.get('safety_violations', 0))
        constraint_violations_rate.append(info.get('constraint_violations', 0))
        
        if episode % 200 == 0:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_success = success_rate[-100:] if len(success_rate) >= 100 else success_rate
            print(f"Episode {episode}, Avg Reward: {np.mean(recent_rewards):.2f}, "
                  f"Success Rate: {np.mean(recent_success)*100:.1f}%, Epsilon: {epsilon:.3f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Calculate final metrics
    final_100_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    final_100_success = success_rate[-100:] if len(success_rate) >= 100 else success_rate
    
    results = {
        'model_name': model_config['name'],
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'invalid_actions_rate': invalid_actions_rate,
        'safety_violations_rate': safety_violations_rate,
        'constraint_violations_rate': constraint_violations_rate,
        'final_avg_reward': np.mean(final_100_rewards),
        'final_success_rate': np.mean(final_100_success),
        'training_time': training_time,
        'total_invalid_actions': sum(invalid_actions_rate),
        'total_safety_violations': sum(safety_violations_rate),
        'total_constraint_violations': sum(constraint_violations_rate)
    }
    
    return results

def run_ablation_experiments():
    """Run all 5 ablation experiments"""
    
    # Define the 5 model configurations
    model_configs = [
        {
            'name': 'Vanilla DQN',
            'use_action_mask': False,
            'use_safety_mechanism': False,
            'use_constraint_check': False
        },
        {
            'name': 'DQN with Action Masking',
            'use_action_mask': True,
            'use_safety_mechanism': False,
            'use_constraint_check': False
        },
        {
            'name': 'DQN with Safety Mechanism',
            'use_action_mask': False,
            'use_safety_mechanism': True,
            'use_constraint_check': False
        },
        {
            'name': 'DQN with Constraint Checking',
            'use_action_mask': False,
            'use_safety_mechanism': False,
            'use_constraint_check': True
        },
        {
            'name': 'Complete Intelligent Agent',
            'use_action_mask': True,
            'use_safety_mechanism': True,
            'use_constraint_check': True
        }
    ]
    
    all_results = []
    
    # Run each model configuration
    for config in model_configs:
        results = train_ablation_model(config, num_episodes=1000, seed=42)
        all_results.append(results)
        print(f"\n{config['name']} Results:")
        print(f"Final Average Reward: {results['final_avg_reward']:.2f}")
        print(f"Final Success Rate: {results['final_success_rate']*100:.1f}%")
        print(f"Training Time: {results['training_time']:.2f}s")
        print(f"Total Invalid Actions: {results['total_invalid_actions']}")
        print(f"Total Safety Violations: {results['total_safety_violations']}")
        print(f"Total Constraint Violations: {results['total_constraint_violations']}")
        print("-" * 60)
    
    return all_results

def create_ablation_visualizations(all_results):
    """Create comprehensive visualization of ablation experiment results"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create results directory
    os.makedirs('ablation_results', exist_ok=True)
    
    # 1. Training Curves Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ablation Study: Training Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    for result in all_results:
        # Smooth the rewards using moving average
        rewards = result['episode_rewards']
        window_size = 50
        smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
        ax1.plot(smoothed_rewards, label=result['model_name'], linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (50-episode window)')
    ax1.set_title('Learning Curves')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Success Rate
    ax2 = axes[0, 1]
    for result in all_results:
        success_rate = result['success_rate']
        window_size = 50
        smoothed_success = pd.Series(success_rate).rolling(window=window_size, min_periods=1).mean()
        ax2.plot(smoothed_success, label=result['model_name'], linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (50-episode window)')
    ax2.set_title('Success Rate Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Final Performance Comparison
    ax3 = axes[1, 0]
    model_names = [result['model_name'] for result in all_results]
    final_rewards = [result['final_avg_reward'] for result in all_results]
    
    bars = ax3.bar(model_names, final_rewards, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Final Average Reward')
    ax3.set_title('Final Performance Comparison (Last 100 Episodes)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_rewards):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Violation Analysis
    ax4 = axes[1, 1]
    violation_data = {
        'Model': model_names,
        'Invalid Actions': [result['total_invalid_actions'] for result in all_results],
        'Safety Violations': [result['total_safety_violations'] for result in all_results],
        'Constraint Violations': [result['total_constraint_violations'] for result in all_results]
    }
    
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    ax4.bar(x_pos - width, violation_data['Invalid Actions'], width, label='Invalid Actions', alpha=0.8)
    ax4.bar(x_pos, violation_data['Safety Violations'], width, label='Safety Violations', alpha=0.8)
    ax4.bar(x_pos + width, violation_data['Constraint Violations'], width, label='Constraint Violations', alpha=0.8)
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Total Violations')
    ax4.set_title('Violation Analysis')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ablation_results/ablation_training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Summary Statistics Table
    summary_df = pd.DataFrame({
        'Model': [result['model_name'] for result in all_results],
        'Final Avg Reward': [f"{result['final_avg_reward']:.2f}" for result in all_results],
        'Success Rate (%)': [f"{result['final_success_rate']*100:.1f}" for result in all_results],
        'Training Time (s)': [f"{result['training_time']:.1f}" for result in all_results],
        'Invalid Actions': [result['total_invalid_actions'] for result in all_results],
        'Safety Violations': [result['total_safety_violations'] for result in all_results],
        'Constraint Violations': [result['total_constraint_violations'] for result in all_results]
    })
    
    # Save summary table
    summary_df.to_csv('ablation_results/ablation_summary.csv', index=False)
    print("\nAblation Study Summary:")
    print(summary_df.to_string(index=False))
    
    # 3. Component Contribution Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Component Contribution Analysis', fontsize=16, fontweight='bold')
    
    # Performance improvement analysis
    baseline_reward = all_results[0]['final_avg_reward']  # Vanilla DQN baseline
    improvements = [result['final_avg_reward'] - baseline_reward for result in all_results]
    
    ax1.bar(model_names, improvements, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Reward Improvement over Baseline')
    ax1.set_title('Performance Improvement Analysis')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, improvement in enumerate(improvements):
        ax1.text(i, improvement + (max(improvements) * 0.01), f'{improvement:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Success rate comparison
    success_rates = [result['final_success_rate'] * 100 for result in all_results]
    bars = ax2.bar(model_names, success_rates, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Safety Performance Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 105)
    
    # Add value labels
    for bar, value in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('ablation_results/component_contribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_df

if __name__ == "__main__":
    print("Starting Ablation Experiments...")
    print("=" * 60)
    
    # Run all ablation experiments
    results = run_ablation_experiments()
    
    # Create visualizations
    print("\nCreating visualizations...")
    summary_table = create_ablation_visualizations(results)
    
    print(f"\nAblation study completed!")
    print(f"Results saved in 'ablation_results/' directory")
    print("=" * 60)