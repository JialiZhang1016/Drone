# corrected_ablation_experiment.py
# 修正后的消融实验：明确的职责划分

# ============================================================================
# 超参数配置区域
# ============================================================================

# 训练参数
TRAINING_PARAMS = {
    'num_episodes': 2000,           # 训练回合数 (增加以获得更稳定的结果)
    'epsilon_start': 1.0,           # 初始探索率
    'epsilon_end': 0.02,            # 最终探索率 (稍微提高以保持探索)
    'epsilon_decay': 0.995,        # 探索率衰减 (更缓慢的衰减)
    'target_update_freq': 100,       # 目标网络更新频率 (更频繁更新)
    'batch_size': 64,               # 批次大小 (增加以提高稳定性)
    'learning_rate': 0.001,        # 学习率 (稍微降低以提高稳定性)
    'memory_capacity': 10000,       # 经验回放池容量 (增加以存储更多经验)
}

# 评价参数
EVALUATION_PARAMS = {
    'window_size': 50,             # 平滑窗口大小
    'final_eval_window': 100,       # 最终评价窗口大小 (增加以获得更稳定的评估)
    'log_print_interval': 500,      # 日志打印间隔 (更频繁的日志)
}

# 可视化参数
VISUALIZATION_PARAMS = {
    'plot_window_size': 50,         # 绘图平滑窗口 (减少以显示更多细节)
    'figure_size': (24, 16),        # 图表大小 (稍微增加高度)
    'dpi': 300,                     # 图片分辨率
}

# ============================================================================
# 导入库
# ============================================================================

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
import sys

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

def train_model_with_clear_responsibilities(model_config, config_file, num_episodes=None, seed=42):
    """
    使用明确职责划分的训练函数
    
    职责划分：
    - 环境(PureDroneRoutePlanningEnv): 纯粹的物理环境，只负责状态转换和奖励
    - Agent(IntelligentDQNAgent): 所有智能决策，包括约束检查、安全机制、动作掩码
    """
    set_random_seed(seed)
    
    # 使用全局超参数，如果提供了num_episodes则覆盖
    if num_episodes is None:
        num_episodes = TRAINING_PARAMS['num_episodes']
    
    # Load environment configuration
    with open(config_file, 'r') as f:
        env_config = json.load(f)
    
    # 创建纯粹的物理环境（不做任何智能决策）
    # 从模型配置中获取 reward shaping 的设置
    use_shaping = model_config.get('use_reward_shaping', True)  # 默认为True以兼容旧配置
    env = PureDroneRoutePlanningEnv(env_config, use_reward_shaping=use_shaping)
    
    # 创建智能Agent（负责所有智能决策）
    agent = IntelligentDQNAgent(
        env,
        use_action_mask=model_config['use_action_mask'],
        use_safety_mechanism=model_config['use_safety_mechanism'],
        use_constraint_check=model_config['use_constraint_check']
    )
    
    # 使用全局超参数
    epsilon_start = TRAINING_PARAMS['epsilon_start']
    epsilon_end = TRAINING_PARAMS['epsilon_end']
    epsilon_decay = TRAINING_PARAMS['epsilon_decay']
    target_update_freq = TRAINING_PARAMS['target_update_freq']
    batch_size = TRAINING_PARAMS['batch_size']
    
    # Experience replay
    memory = ExperienceReplay(capacity=TRAINING_PARAMS['memory_capacity'])
    
    # Training metrics
    episode_rewards = []
    episode_base_rewards = []   # 新增：记录每回合的核心任务奖励
    episode_shaping_rewards = [] # 新增：记录每回合的塑形奖励
    episode_lengths = []
    success_rates = []
    episode_epsilons = []       # 新增：记录每回合的探索率
    
    # 使用全局评价参数
    window_size = EVALUATION_PARAMS['window_size']
    final_N = EVALUATION_PARAMS['final_eval_window']
    
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
        total_base_reward = 0.0
        total_shaping_reward = 0.0
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
            
            # Store experience (Agent 学习仍然使用总奖励 `reward`)
            memory.push(state, action_idx, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            total_base_reward += info.get('reward_base', 0)  # 从info字典中累积
            total_shaping_reward += info.get('reward_shaping', 0)  # 从info字典中累积
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
        episode_base_rewards.append(total_base_reward)
        episode_shaping_rewards.append(total_shaping_reward)
        episode_lengths.append(episode_length)
        episode_epsilons.append(epsilon)  # 记录当前探索率
        
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
        
        if episode % EVALUATION_PARAMS['log_print_interval'] == 0:
            recent_rewards = episode_rewards[-window_size:] if len(episode_rewards) >= window_size else episode_rewards
            recent_success = success_rates[-window_size:] if len(success_rates) >= window_size else success_rates
            print(f"Episode {episode}, Avg Reward: {np.mean(recent_rewards):.2f}, "
                  f"Success Rate: {np.mean(recent_success)*100:.1f}%, Epsilon: {epsilon:.3f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Calculate final metrics
    final_100_rewards = episode_rewards[-final_N:] if len(episode_rewards) >= final_N else episode_rewards
    final_100_base_rewards = episode_base_rewards[-final_N:] if len(episode_base_rewards) >= final_N else episode_base_rewards
    final_100_shaping_rewards = episode_shaping_rewards[-final_N:] if len(episode_shaping_rewards) >= final_N else episode_shaping_rewards
    final_100_success = success_rates[-final_N:] if len(success_rates) >= final_N else success_rates
    
    results = {
        'model_name': model_config['name'],
        'model_config': model_config,
        'config_file': config_file,
        
        # 基础性能指标
        'episode_rewards': episode_rewards,
        'episode_base_rewards': episode_base_rewards,  # 添加到结果
        'episode_shaping_rewards': episode_shaping_rewards,  # 添加到结果
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'episode_epsilons': episode_epsilons,  # 新增：探索率历史
        'final_avg_reward': np.mean(final_100_rewards),
        'final_avg_base_reward': np.mean(final_100_base_rewards),  # 添加到结果
        'final_avg_shaping_reward': np.mean(final_100_shaping_rewards),  # 添加到结果
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

def run_ablation_experiments(config_file):
    """运行修正后的消融实验"""
    
    # 重新定义6个模型配置（增加reward shaping控制）
    model_configs = [
        {
            'name': '1. Vanilla DQN',
            'description': 'No intelligent components, direct RL on physical environment',
            'use_action_mask': False,
            'use_safety_mechanism': False,
            'use_constraint_check': False,
            'use_reward_shaping': False # No Shaping
        },
        {
            'name': '2. DQN + Shaping', # 新增：专门测试Shaping效果
            'description': 'DQN with only Reward Shaping enabled',
            'use_action_mask': False,
            'use_safety_mechanism': False,
            'use_constraint_check': False,
            'use_reward_shaping': True # Only Shaping
        },
        {
            'name': '3. DQN + Action Masking',
            'description': 'DQN with action masking to filter invalid actions',
            'use_action_mask': True,
            'use_safety_mechanism': False,
            'use_constraint_check': False,
            'use_reward_shaping': False # No Shaping
        },
        {
            'name': '4. Complete Agent', # 最终版
            'description': 'All intelligent components enabled with reward shaping',
            'use_action_mask': True,
            'use_safety_mechanism': False,
            'use_constraint_check': False,
            'use_reward_shaping': True # With Shaping
        }
    ]
    
    all_results = []
    
    # 从配置文件名称中提取location数量
    config_name = os.path.basename(config_file).replace('.json', '')
    num_locations = config_name.split('_')[-1]
    
    print("=" * 80)
    print(f"ABLATION STUDY: {num_locations} Locations Configuration")
    print("Environment: Pure physical simulation (no intelligence)")
    print("Agent: All intelligent decision making")
    print(f"Config File: {config_file}")
    print("=" * 80)
    
    # Run each model configuration
    for config in model_configs:
        print(f"\n{'-'*60}")
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Components: Action_Mask={config['use_action_mask']}, "
              f"Safety={config['use_safety_mechanism']}, "
              f"Constraint={config['use_constraint_check']}, "
              f"Reward_Shaping={config['use_reward_shaping']}")
        print(f"{'-'*60}")
        
        results = train_model_with_clear_responsibilities(config, config_file, num_episodes=None, seed=42)
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
    
    return all_results, num_locations

def create_ablation_visualizations(all_results, num_locations, output_dir='ablation_results', config_file=None, timestamp=None):
    """创建修正后的消融实验可视化"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 主要性能对比图
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))  # 修正为2x3布局
    fig.suptitle(f'Ablation Study: {num_locations} Locations - Component Contribution Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Learning Curves (现在展示任务核心收益)
    ax1 = axes[0, 0]
    for result in all_results:
        # 使用 'episode_base_rewards' 进行绘图
        rewards = result['episode_base_rewards']
        plot_window_size = VISUALIZATION_PARAMS['plot_window_size']
        smoothed_rewards = pd.Series(rewards).rolling(window=plot_window_size, min_periods=1).mean()
        ax1.plot(smoothed_rewards, label=result['model_name'].split(':')[0], linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(f'Average Task Reward ({plot_window_size}-episode window)')
    ax1.set_title('Learning Performance (Core Task Reward)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Success Rate Evolution
    ax2 = axes[0, 1]
    for result in all_results:
        success_rates = result['success_rates']
        plot_window_size = VISUALIZATION_PARAMS['plot_window_size']
        smoothed_success = pd.Series(success_rates).rolling(window=plot_window_size, min_periods=1).mean()
        ax2.plot(smoothed_success, label=result['model_name'].split(':')[0], linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel(f'Success Rate ({plot_window_size}-episode window)')
    ax2.set_title('Safety Performance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Exploration Rate Evolution (新增)
    ax3 = axes[0, 2]
    for result in all_results:
        epsilons = result['episode_epsilons']
        ax3.plot(epsilons, label=result['model_name'].split(':')[0], linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon (Exploration Rate)')
    ax3.set_title('Exploration Rate Decay')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale('log')  # 使用对数刻度更好地显示衰减
    
    # Plot 4: Final Performance Comparison (现在展示最终的核心任务收益)
    ax4 = axes[1, 0]
    model_names = [result['model_name'].split(':')[0] for result in all_results]
    # 使用 'final_avg_base_reward' 进行绘图
    final_rewards = [result['final_avg_base_reward'] for result in all_results]
    
    bars = ax4.bar(model_names, final_rewards, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Final Average Task Reward')  # 标签更新
    ax4.set_title('Final Task Performance Comparison')  # 标题更新
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, final_rewards):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(final_rewards)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 5: Component Activity Analysis (智能组件工作统计)
    ax5 = axes[1, 1]
    
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
    ax5.bar(x_pos - width/2, env_violation_data['Env Safety Violations'], width, 
            label='Environment Safety Violations', alpha=0.8, color='red')
    ax5.bar(x_pos + width/2, env_violation_data['Env Constraint Violations'], width, 
            label='Environment Constraint Violations', alpha=0.8, color='orange')
    
    ax5.set_xlabel('Model')
    ax5.set_ylabel('Environment Violations Count')
    ax5.set_title('Actual Environment Violations')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(model_names, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Final Exploration Rate Comparison (新增)
    ax6 = axes[1, 2]
    final_epsilons = [result['episode_epsilons'][-1] for result in all_results]
    
    bars = ax6.bar(model_names, final_epsilons, alpha=0.8, edgecolor='black')
    ax6.set_ylabel('Final Epsilon Value')
    ax6.set_title('Final Exploration Rate Comparison')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, final_epsilons):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(final_epsilons)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_analysis_{num_locations}locations_V2.png', 
                dpi=VISUALIZATION_PARAMS['dpi'], bbox_inches='tight')
    
    # 2. 创建详细的组件贡献分析表 (添加分解后的奖励列)
    analysis_df = pd.DataFrame({
        'Model': [result['model_name'] for result in all_results],
        'Action Mask': [result['model_config']['use_action_mask'] for result in all_results],
        'Safety Mechanism': [result['model_config']['use_safety_mechanism'] for result in all_results],
        'Constraint Check': [result['model_config']['use_constraint_check'] for result in all_results],
        'Reward Shaping': [result['model_config']['use_reward_shaping'] for result in all_results],
        'Final Base Reward': [f"{result['final_avg_base_reward']:.2f}" for result in all_results],  # 核心指标
        'Final Shaping Reward': [f"{result['final_avg_shaping_reward']:.2f}" for result in all_results],  # 诊断指标
        'Final Total Reward': [f"{result['final_avg_reward']:.2f}" for result in all_results],  # 参考指标
        'Success Rate (%)': [f"{result['final_success_rate']*100:.1f}" for result in all_results],
        'Training Time (s)': [f"{result['training_time']:.1f}" for result in all_results],
        'Agent Mask Work': [result['total_action_mask_rejections'] for result in all_results],
        'Agent Safety Work': [result['total_safety_interventions'] for result in all_results],
        'Env Safety Violations': [result['total_env_safety_violations'] for result in all_results],
        'Env Constraint Violations': [result['total_env_constraint_violations'] for result in all_results]
    })
    
    # Save analysis table
    analysis_df.to_csv(f'{output_dir}/ablation_analysis_{num_locations}locations_V2.csv', index=False)
    
    # Save configuration and parameters
    import shutil
    import json
    
    # 保存config文件
    config_file_name = os.path.basename(config_file)
    shutil.copy2(config_file, f'{output_dir}/{config_file_name}')
    
    # 保存完整的实验配置
    experiment_config = {
        'experiment_info': {
            'num_locations': num_locations,
            'config_file': config_file,
            'timestamp': timestamp,
            'num_episodes': TRAINING_PARAMS['num_episodes']
        },
        'training_params': TRAINING_PARAMS,
        'evaluation_params': EVALUATION_PARAMS,
        'visualization_params': VISUALIZATION_PARAMS,
        'model_configs': [
            {
                'name': '1. Vanilla DQN',
                'description': 'No intelligent components, direct RL on physical environment',
                'use_action_mask': False,
                'use_safety_mechanism': False,
                'use_constraint_check': False,
                'use_reward_shaping': False
            },
            {
                'name': '2. DQN + Shaping',
                'description': 'DQN with only Reward Shaping enabled',
                'use_action_mask': False,
                'use_safety_mechanism': False,
                'use_constraint_check': False,
                'use_reward_shaping': True
            },
            {
                'name': '3. DQN + Action Masking',
                'description': 'DQN with action masking to filter invalid actions',
                'use_action_mask': True,
                'use_safety_mechanism': False,
                'use_constraint_check': False,
                'use_reward_shaping': False
            },
            {
                'name': '4. Complete Agent',
                'description': 'All intelligent components enabled with reward shaping',
                'use_action_mask': True,
                'use_safety_mechanism': False,
                'use_constraint_check': False,
                'use_reward_shaping': True
            }
        ]
    }
    
    with open(f'{output_dir}/experiment_config.json', 'w') as f:
        json.dump(experiment_config, f, indent=4)
    
    # 创建README文件
    readme_content = f"""# Ablation Study Experiment Results

## Experiment Information
- **Date**: {timestamp}
- **Locations**: {num_locations}
- **Episodes**: {TRAINING_PARAMS['num_episodes']}
- **Config File**: {config_file}

## Files Description
- `experiment_config.json`: Complete experiment configuration including all parameters
- `{config_file_name}`: Original environment configuration file
- `ablation_analysis_{num_locations}locations_V2.csv`: Detailed results table
- `ablation_analysis_{num_locations}locations_V2.png`: Visualization charts

## Model Configurations
1. **Vanilla DQN**: No intelligent components
2. **DQN + Shaping**: Only reward shaping enabled
3. **DQN + Action Masking**: Action masking to filter invalid actions
4. **Complete Agent**: All intelligent components enabled

## Key Parameters
- **Training Episodes**: {TRAINING_PARAMS['num_episodes']}
- **Epsilon Decay**: {TRAINING_PARAMS['epsilon_decay']}
- **Batch Size**: {TRAINING_PARAMS['batch_size']}
- **Evaluation Window**: {EVALUATION_PARAMS['final_eval_window']}
- **Log Interval**: {EVALUATION_PARAMS['log_print_interval']}
"""
    
    with open(f'{output_dir}/README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"\n" + "="*80)
    print(f"ABLATION STUDY RESULTS (V2): {num_locations} Locations")
    print("="*80)
    print(analysis_df.to_string(index=False))
    
    
    return analysis_df

def run_single_experiment(config_file):
    """运行单个配置文件的消融实验"""
    print(f"Starting Ablation Experiments for {config_file}...")
    
    # 创建带时间戳和位置信息的输出目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    
    # 获取训练步数
    num_episodes = TRAINING_PARAMS['num_episodes']  # 从全局参数中获取
    
    # 提取位置数量
    config_name = os.path.basename(config_file).replace('.json', '')
    num_locations = config_name.split('_')[-1]
    
    # 创建文件夹名称
    output_dir = f"ablation_results/{timestamp}_{num_episodes}eposid_{num_locations}"
    
    # Run experiments
    results, num_locations = run_ablation_experiments(config_file)
    
    # Create visualizations and analysis
    print(f"\nCreating analysis and visualizations for {num_locations} locations...")
    analysis_table = create_ablation_visualizations(results, num_locations, output_dir, config_file, timestamp)
    
    print(f"\nAblation study for {num_locations} locations completed!")
    print(f"Results saved in '{output_dir}' directory")
    print("="*80)
    
    return results, analysis_table

def run_all_configurations():
    """运行所有三个配置文件的消融实验"""
    config_files = ['config/config_5.json', 'config/config_8.json', 'config/config_10.json']
    
    # 创建带时间戳和位置信息的输出目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    
    # 获取训练步数
    num_episodes = TRAINING_PARAMS['num_episodes']  # 从全局参数中获取
    
    # 提取位置数量
    location_numbers = []
    for config_file in config_files:
        if os.path.exists(config_file):
            config_name = os.path.basename(config_file).replace('.json', '')
            num_locations = config_name.split('_')[-1]
            location_numbers.append(num_locations)
    
    # 创建文件夹名称
    locations_str = "_".join(location_numbers)
    output_dir = f"ablation_results/{timestamp}_{num_episodes}eposid_{locations_str}"
    
    all_results_summary = {}
    
    print("="*80)
    print("RUNNING ABLATION EXPERIMENTS FOR ALL CONFIGURATIONS")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    
    for i, config_file in enumerate(config_files, 1):
        if os.path.exists(config_file):
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i}/3: {config_file}")
            print(f"{'='*80}")
            
            try:
                # 为每个配置创建单独的结果
                results, num_locations = run_ablation_experiments(config_file)
                
                # 创建可视化并保存到指定目录
                analysis_table = create_ablation_visualizations(results, num_locations, output_dir, config_file, timestamp)
                
                # 保存每个配置的结果摘要
                config_name = os.path.basename(config_file).replace('.json', '')
                all_results_summary[config_name] = {
                    'results': results,
                    'analysis_table': analysis_table,
                    'config_file': config_file
                }
                
                print(f"\n✓ Experiment {i}/3 completed successfully!")
                
            except Exception as e:
                print(f"\n✗ Experiment {i}/3 failed with error: {e}")
                continue
        else:
            print(f"\n✗ Config file {config_file} not found, skipping...")
    
    # 创建跨配置的对比分析
    if len(all_results_summary) > 1:
        create_cross_config_comparison(all_results_summary, output_dir)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved in '{output_dir}' directory")
    print("="*80)
    
    return all_results_summary

def create_cross_config_comparison(all_results_summary, output_dir='ablation_results'):
    """创建跨配置的对比分析"""
    print(f"\nCreating cross-configuration comparison...")
    
    # 创建跨配置对比图
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # 使用更合适的2x2布局大小
    fig.suptitle('Cross-Configuration Ablation Study Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 提取数据
    config_names = []
    final_rewards = []
    success_rates = []
    training_times = []
    env_safety_violations = []
    
    # 动态获取所有模型名称
    all_model_names = set()
    for config_name, data in all_results_summary.items():
        results = data['results']
        for result in results:
            all_model_names.add(result['model_name'])
    
    # 转换为列表并排序，确保一致性
    model_names = sorted(list(all_model_names))
    max_models = len(model_names)
    
    for config_name, data in all_results_summary.items():
        config_names.append(config_name)
        results = data['results']
        
        # 收集每个模型的最终性能
        config_final_rewards = []
        config_success_rates = []
        config_training_times = []
        config_env_safety_violations = []
        
        # 创建模型名称到索引的映射
        model_to_index = {name: i for i, name in enumerate(model_names)}
        
        # 初始化数组，用NaN填充缺失的模型
        config_final_rewards = [np.nan] * max_models
        config_success_rates = [np.nan] * max_models
        config_training_times = [np.nan] * max_models
        config_env_safety_violations = [np.nan] * max_models
        
        for result in results:
            model_name = result['model_name']
            if model_name in model_to_index:
                idx = model_to_index[model_name]
                config_final_rewards[idx] = result['final_avg_reward']
                config_success_rates[idx] = result['final_success_rate'] * 100
                config_training_times[idx] = result['training_time']
                config_env_safety_violations[idx] = result['total_env_safety_violations']
        
        final_rewards.append(config_final_rewards)
        success_rates.append(config_success_rates)
        training_times.append(config_training_times)
        env_safety_violations.append(config_env_safety_violations)
    
    # Plot 1: Final Rewards Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, config_name in enumerate(config_names):
        # 过滤掉NaN值
        valid_data = [(j, val) for j, val in enumerate(final_rewards[i]) if not np.isnan(val)]
        if valid_data:
            valid_indices = [item[0] for item in valid_data]
            valid_values = [item[1] for item in valid_data]
            ax1.bar(x[valid_indices] + i*width, valid_values, width, label=config_name, alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Final Average Reward')
    ax1.set_title('Final Performance Across Configurations')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Success Rates Comparison
    ax2 = axes[0, 1]
    for i, config_name in enumerate(config_names):
        # 过滤掉NaN值
        valid_data = [(j, val) for j, val in enumerate(success_rates[i]) if not np.isnan(val)]
        if valid_data:
            valid_indices = [item[0] for item in valid_data]
            valid_values = [item[1] for item in valid_data]
            ax2.bar(x[valid_indices] + i*width, valid_values, width, label=config_name, alpha=0.8)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Safety Performance Across Configurations')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Training Time Comparison
    ax3 = axes[1, 0]
    for i, config_name in enumerate(config_names):
        # 过滤掉NaN值
        valid_data = [(j, val) for j, val in enumerate(training_times[i]) if not np.isnan(val)]
        if valid_data:
            valid_indices = [item[0] for item in valid_data]
            valid_values = [item[1] for item in valid_data]
            ax3.bar(x[valid_indices] + i*width, valid_values, width, label=config_name, alpha=0.8)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Efficiency Across Configurations')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Environment Safety Violations Comparison
    ax4 = axes[1, 1]
    for i, config_name in enumerate(config_names):
        # 过滤掉NaN值
        valid_data = [(j, val) for j, val in enumerate(env_safety_violations[i]) if not np.isnan(val)]
        if valid_data:
            valid_indices = [item[0] for item in valid_data]
            valid_values = [item[1] for item in valid_data]
            ax4.bar(x[valid_indices] + i*width, valid_values, width, label=config_name, alpha=0.8)
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Environment Safety Violations')
    ax4.set_title('Safety Violations Across Configurations')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cross_config_comparison_V2.png', 
                dpi=VISUALIZATION_PARAMS['dpi'], bbox_inches='tight')
    
    # 创建跨配置对比表
    comparison_data = []
    for config_name, data in all_results_summary.items():
        results = data['results']
        for i, result in enumerate(results):
            comparison_data.append({
                'Configuration': config_name,
                'Model': result['model_name'],
                'Final Reward': f"{result['final_avg_reward']:.2f}",
                'Success Rate (%)': f"{result['final_success_rate']*100:.1f}",
                'Training Time (s)': f"{result['training_time']:.1f}",
                'Env Safety Violations': result['total_env_safety_violations'],
                'Env Constraint Violations': result['total_env_constraint_violations'],
                'Action Mask Rejections': result['total_action_mask_rejections'],
                'Safety Interventions': result['total_safety_interventions']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f'{output_dir}/cross_config_comparison_V2.csv', index=False)
    
    print(f"\nCross-configuration comparison saved:")
    print(f"  - Chart: {output_dir}/cross_config_comparison_V2.png")
    print(f"  - Data: {output_dir}/cross_config_comparison_V2.csv")
    
    return comparison_df

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"Error: Config file {config_file} not found!")
            sys.exit(1)
        run_single_experiment(config_file)
    else:
        # 默认运行所有三个配置
        run_all_configurations()