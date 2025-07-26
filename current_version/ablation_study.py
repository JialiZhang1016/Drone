# corrected_ablation_experiment.py
# 修正后的消融实验：明确的职责划分

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

from drone_env import PureDroneRoutePlanningEnv
from agent.dqn_agent_ablation import IntelligentDQNAgent

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

def train_model_with_clear_responsibilities(model_config, num_episodes=2000, seed=42):
    """
    使用明确职责划分的训练函数
    
    职责划分：
    - 环境(PureDroneRoutePlanningEnv): 纯粹的物理环境，只负责状态转换和奖励
    - Agent(IntelligentDQNAgent): 所有智能决策，包括约束检查、安全机制、动作掩码
    """
    set_random_seed(seed)
    
    # Load environment configuration
    with open('config/realword_8/config_8.json', 'r') as f:
        env_config = json.load(f)
    
    # 创建纯粹的物理环境（不做任何智能决策）
    env = PureDroneRoutePlanningEnv(env_config)
    
    # 创建智能Agent（负责所有智能决策）
    agent = IntelligentDQNAgent(
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
    success_rates = []
    
    # Agent-specific metrics (智能决策统计)
    action_mask_rejections_per_episode = []
    safety_interventions_per_episode = []
    constraint_violations_per_episode = []
    
    # Environment-specific metrics (环境统计)
    env_safety_violations_per_episode = []  # 环境中实际发生的安全违规
    env_constraint_violations_per_episode = []  # 环境中实际发生的约束违规
    
    epsilon = epsilon_start
    
    print(f"Training {model_config['name']}...")
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.reset_statistics()  # 重置Agent统计
        
        total_reward = 0
        episode_length = 0
        
        while True:
            # Agent选择动作（包含所有智能决策）
            action_idx = agent.select_action(state, epsilon)
            
            if action_idx is None:
                # Agent判断无有效动作，结束回合
                break
            
            action = agent.action_index_mapping[action_idx]
            
            # 环境执行动作（纯物理状态转换）
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
        
        # 成功率：安全返回家
        if 'next_state' in info:
            current_location = info['next_state']['current_location']
        else:
            current_location = state['current_location']
        success = (current_location == 0)
        success_rates.append(1 if success else 0)
        
        # Agent智能决策统计
        agent_stats = agent.get_statistics()
        action_mask_rejections_per_episode.append(agent_stats['action_mask_rejections'])
        safety_interventions_per_episode.append(agent_stats['safety_interventions'])
        constraint_violations_per_episode.append(agent_stats['constraint_violations'])
        
        # 环境违规统计（从环境获取）
        env_safety_violations_per_episode.append(env.safety_violations)
        env_constraint_violations_per_episode.append(env.constraint_violations)
        
        if episode % 200 == 0:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_success = success_rates[-100:] if len(success_rates) >= 100 else success_rates
            print(f"Episode {episode}, Avg Reward: {np.mean(recent_rewards):.2f}, "
                  f"Success Rate: {np.mean(recent_success)*100:.1f}%, Epsilon: {epsilon:.3f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Calculate final metrics
    final_100_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    final_100_success = success_rates[-100:] if len(success_rates) >= 100 else success_rates
    
    results = {
        'model_name': model_config['name'],
        'model_config': model_config,
        
        # 基础性能指标
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'final_avg_reward': np.mean(final_100_rewards),
        'final_success_rate': np.mean(final_100_success),
        'training_time': training_time,
        
        # Agent智能决策统计（显示各组件的工作情况）
        'action_mask_rejections_per_episode': action_mask_rejections_per_episode,
        'safety_interventions_per_episode': safety_interventions_per_episode,
        'constraint_violations_per_episode': constraint_violations_per_episode,
        'total_action_mask_rejections': sum(action_mask_rejections_per_episode),
        'total_safety_interventions': sum(safety_interventions_per_episode),
        'total_constraint_violations': sum(constraint_violations_per_episode),
        
        # 环境违规统计（显示实际的物理问题）
        'env_safety_violations_per_episode': env_safety_violations_per_episode,
        'env_constraint_violations_per_episode': env_constraint_violations_per_episode,
        'total_env_safety_violations': sum(env_safety_violations_per_episode),
        'total_env_constraint_violations': sum(env_constraint_violations_per_episode),
    }
    
    return results

def run_ablation_experiments():
    """运行修正后的消融实验"""
    
    # 重新定义5个模型配置
    model_configs = [
        {
            'name': 'Vanilla DQN',
            'description': 'No intelligent components, direct RL on physical environment',
            'use_action_mask': False,
            'use_safety_mechanism': False,
            'use_constraint_check': False
        },
        {
            'name': 'DQN with Action Masking',
            'description': 'Add action masking to filter invalid actions',
            'use_action_mask': True,
            'use_safety_mechanism': False,
            'use_constraint_check': False
        },
        {
            'name': 'DQN with Safety Mechanism',
            'description': 'Add 20% safety margin forced return home',
            'use_action_mask': False,
            'use_safety_mechanism': True,
            'use_constraint_check': False
        },
        {
            'name': 'DQN with Constraint Checking',
            'description': 'Add time constraint checking',
            'use_action_mask': False,
            'use_safety_mechanism': False,
            'use_constraint_check': True
        },
        {
            'name': 'Complete Intelligent Agent',
            'description': 'All intelligent components enabled',
            'use_action_mask': True,
            'use_safety_mechanism': True,
            'use_constraint_check': True
        }
    ]
    
    all_results = []
    
    print("=" * 80)
    print("CORRECTED ABLATION STUDY: Clear Responsibility Division")
    print("Environment: Pure physical simulation (no intelligence)")
    print("Agent: All intelligent decision making")
    print("=" * 80)
    
    # Run each model configuration
    for config in model_configs:
        print(f"\n{'-'*60}")
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Components: Action_Mask={config['use_action_mask']}, "
              f"Safety={config['use_safety_mechanism']}, "
              f"Constraint={config['use_constraint_check']}")
        print(f"{'-'*60}")
        
        results = train_model_with_clear_responsibilities(config, num_episodes=1500, seed=42)
        all_results.append(results)
        
        print(f"\nResults for {config['name']}:")
        print(f"  Final Average Reward: {results['final_avg_reward']:.2f}")
        print(f"  Final Success Rate: {results['final_success_rate']*100:.1f}%")
        print(f"  Training Time: {results['training_time']:.2f}s")
        print(f"  Agent Decision Statistics:")
        print(f"    Action Mask Rejections: {results['total_action_mask_rejections']}")
        print(f"    Safety Interventions: {results['total_safety_interventions']}")
        print(f"    Constraint Violations: {results['total_constraint_violations']}")
        print(f"  Environment Statistics:")
        print(f"    Safety Violations: {results['total_env_safety_violations']}")
        print(f"    Constraint Violations: {results['total_env_constraint_violations']}")
    
    return all_results

def create_ablation_visualizations(all_results):
    """创建修正后的消融实验可视化"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create results directory
    os.makedirs('ablation_results', exist_ok=True)
    
    # 1. 主要性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ablation Study: Component Contribution Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Learning Curves
    ax1 = axes[0, 0]
    for result in all_results:
        rewards = result['episode_rewards']
        window_size = 50
        smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
        ax1.plot(smoothed_rewards, label=result['model_name'].split(':')[0], linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (50-episode window)')
    ax1.set_title('Learning Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Success Rate Evolution
    ax2 = axes[0, 1]
    for result in all_results:
        success_rates = result['success_rates']
        window_size = 50
        smoothed_success = pd.Series(success_rates).rolling(window=window_size, min_periods=1).mean()
        ax2.plot(smoothed_success, label=result['model_name'].split(':')[0], linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (50-episode window)')
    ax2.set_title('Safety Performance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Final Performance Comparison
    ax3 = axes[1, 0]
    model_names = [result['model_name'].split(':')[0] for result in all_results]
    final_rewards = [result['final_avg_reward'] for result in all_results]
    
    bars = ax3.bar(model_names, final_rewards, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Final Average Reward')
    ax3.set_title('Final Performance Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, final_rewards):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(final_rewards)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Component Activity Analysis (智能组件工作统计)
    ax4 = axes[1, 1]
    
    component_data = {
        'Model': model_names,
        'Action Mask': [result['total_action_mask_rejections'] for result in all_results],
        'Safety Mechanism': [result['total_safety_interventions'] for result in all_results],
        'Constraint Check': [result['total_constraint_violations'] for result in all_results]
    }
    
    # Environment violations data
    env_violation_data = {
        'Model': model_names,
        'Env Safety Violations': [result['total_env_safety_violations'] for result in all_results],
        'Env Constraint Violations': [result['total_env_constraint_violations'] for result in all_results]
    }
    
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    # Show environment violations instead of component activity
    width = 0.35
    ax4.bar(x_pos - width/2, env_violation_data['Env Safety Violations'], width, 
            label='Environment Safety Violations', alpha=0.8, color='red')
    ax4.bar(x_pos + width/2, env_violation_data['Env Constraint Violations'], width, 
            label='Environment Constraint Violations', alpha=0.8, color='orange')
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Environment Violations Count')
    ax4.set_title('Actual Environment Violations')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ablation_results/ablation_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 创建详细的组件贡献分析表
    analysis_df = pd.DataFrame({
        'Model': [result['model_name'] for result in all_results],
        'Action Mask': [result['model_config']['use_action_mask'] for result in all_results],
        'Safety Mechanism': [result['model_config']['use_safety_mechanism'] for result in all_results],
        'Constraint Check': [result['model_config']['use_constraint_check'] for result in all_results],
        'Final Reward': [f"{result['final_avg_reward']:.2f}" for result in all_results],
        'Success Rate (%)': [f"{result['final_success_rate']*100:.1f}" for result in all_results],
        'Training Time (s)': [f"{result['training_time']:.1f}" for result in all_results],
        'Agent Mask Work': [result['total_action_mask_rejections'] for result in all_results],
        'Agent Safety Work': [result['total_safety_interventions'] for result in all_results],
        'Env Safety Violations': [result['total_env_safety_violations'] for result in all_results],
        'Env Constraint Violations': [result['total_env_constraint_violations'] for result in all_results]
    })
    
    # Save analysis table
    analysis_df.to_csv('ablation_results/ablation_analysis.csv', index=False)
    
    print("\n" + "="*80)
    print("ABALATION STUDY RESULTS")
    print("="*80)
    print(analysis_df.to_string(index=False))
    
    # 3. 组件贡献度分析
    print(f"\nKEY INSIGHTS:")
    print(f"1. Action Masking Contribution:")
    model1_reward = all_results[0]['final_avg_reward']
    model2_reward = all_results[1]['final_avg_reward']
    print(f"   Reward change: {model2_reward - model1_reward:.2f}")
    print(f"   Action mask rejections: {all_results[1]['total_action_mask_rejections']}")
    
    print(f"2. Safety Mechanism Contribution:")
    model3_reward = all_results[2]['final_avg_reward']
    print(f"   Reward change: {model3_reward - model1_reward:.2f}")
    print(f"   Safety interventions: {all_results[2]['total_safety_interventions']}")
    
    print(f"3. Complete System:")
    model5_reward = all_results[4]['final_avg_reward']
    model5_success = all_results[4]['final_success_rate']
    print(f"   Final reward: {model5_reward:.2f}")
    print(f"   Success rate: {model5_success*100:.1f}%")
    print(f"   Environment safety violations: {all_results[4]['total_env_safety_violations']}")
    print(f"   Environment constraint violations: {all_results[4]['total_env_constraint_violations']}")
    
    return analysis_df

if __name__ == "__main__":
    print("Starting Ablation Experiments...")
    
    # Run experiments
    results = run_ablation_experiments()
    
    # Create visualizations and analysis
    print("\nCreating analysis and visualizations...")
    analysis_table = create_ablation_visualizations(results)
    
    print(f"\nAblation study completed!")
    print(f"Results saved in 'ablation_results/' directory")
    print("="*80)