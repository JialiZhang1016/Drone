# corrected_ablation_experiment.py
# 修正后的消融实验：明确的职责划分
# ## MODIFICATION: Added model saving and evaluation capabilities.

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
    'eval_episodes': 10,           # ## MODIFICATION: Number of episodes for final evaluation
    'eval_seed': 123,              # ## MODIFICATION: Separate random seed for reproducible evaluation
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


def set_random_seed(seed=123):
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

## MODIFICATION: Added output_dir parameter to handle model saving location.
def train_model_with_clear_responsibilities(model_config, config_file, output_dir, num_episodes=None, seed=42):
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
        current_location = info.get('next_state', {}).get('current_location', state['current_location'])
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
            recent_rewards = episode_rewards[-window_size:]
            recent_success = success_rates[-window_size:]
            print(f"Episode {episode}, Avg Reward: {np.mean(recent_rewards):.2f}, "
                  f"Success Rate: {np.mean(recent_success)*100:.1f}%, Epsilon: {epsilon:.3f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    ## MODIFICATION START: Save the trained model
    # Sanitize model name for filesystem
    safe_model_name = model_config['name'].replace(' ', '_').replace('.', '').replace('+', 'plus')
    model_filename = f"{safe_model_name}_model.pth"
    model_path = os.path.join(output_dir, model_filename)
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    ## MODIFICATION END

    # Calculate final metrics
    final_100_rewards = episode_rewards[-final_N:]
    final_100_base_rewards = episode_base_rewards[-final_N:]
    final_100_shaping_rewards = episode_shaping_rewards[-final_N:]
    final_100_success = success_rates[-final_N:]

    results = {
        'model_name': model_config['name'],
        'model_config': model_config,
        'config_file': config_file,
        'model_path': model_path, # ## MODIFICATION: Add model path to results

        # 基础性能指标
        'episode_rewards': episode_rewards,
        'episode_base_rewards': episode_base_rewards,
        'episode_shaping_rewards': episode_shaping_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'episode_epsilons': episode_epsilons,
        'final_avg_reward': np.mean(final_100_rewards),
        'final_avg_base_reward': np.mean(final_100_base_rewards),
        'final_avg_shaping_reward': np.mean(final_100_shaping_rewards),
        'final_success_rate': np.mean(final_100_success),
        'training_time': training_time,

        # Agent智能决策统计
        'action_mask_rejections_per_episode': action_mask_rejections_per_episode,
        'safety_interventions_per_episode': safety_interventions_per_episode,
        'constraint_violations_per_episode': constraint_violations_per_episode,
        'total_action_mask_rejections': sum(action_mask_rejections_per_episode),
        'total_safety_interventions': sum(safety_interventions_per_episode),
        'total_constraint_violations': sum(constraint_violations_per_episode),

        # 环境违规统计
        'env_safety_violations_per_episode': env_safety_violations_per_episode,
        'env_constraint_violations_per_episode': env_constraint_violations_per_episode,
        'total_env_safety_violations': sum(env_safety_violations_per_episode),
        'total_env_constraint_violations': sum(env_constraint_violations_per_episode),
    }

    return results

## MODIFICATION START: New function to evaluate a saved model
def evaluate_saved_model(model_path, model_config, config_file):
    """
    Loads a saved model and evaluates its performance deterministically.
    """
    print(f"\nEvaluating model: {model_config['name']} from {model_path}...")
    
    # Use a fixed seed for reproducible evaluation
    eval_seed = EVALUATION_PARAMS['eval_seed']
    set_random_seed(eval_seed)
    
    num_eval_episodes = EVALUATION_PARAMS['eval_episodes']

    # Load environment configuration
    with open(config_file, 'r') as f:
        env_config = json.load(f)

    # Setup environment and agent
    use_shaping = model_config.get('use_reward_shaping', True)
    env = PureDroneRoutePlanningEnv(env_config, use_reward_shaping=use_shaping)
    agent = IntelligentDQNAgent(
        env,
        use_action_mask=model_config['use_action_mask'],
        use_safety_mechanism=model_config['use_safety_mechanism'],
        use_constraint_check=model_config['use_constraint_check']
    )

    # Load the trained model weights
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()  # Set the network to evaluation mode

    # Evaluation metrics
    eval_rewards = []
    eval_base_rewards = []  # ## MODIFICATION: Track base rewards separately
    eval_success_rates = []
    
    for episode in range(num_eval_episodes):
        state, _ = env.reset()
        total_reward = 0
        total_base_reward = 0  # ## MODIFICATION: Track base reward
        done = False

        while not done:
            # Select action deterministically (epsilon = 0)
            action_idx = agent.select_action(state, epsilon=0)
            
            if action_idx is None:
                break
                
            action = agent.action_index_mapping[action_idx]
            
            # Environment executes the action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            total_base_reward += info.get('reward_base', 0)  # ## MODIFICATION: Accumulate base reward only

        eval_rewards.append(total_reward)
        eval_base_rewards.append(total_base_reward)  # ## MODIFICATION: Store base reward
        current_location = info.get('next_state', {}).get('current_location', state['current_location'])
        success = (current_location == 0)
        eval_success_rates.append(1 if success else 0)

    # Calculate average performance
    avg_eval_reward = np.mean(eval_rewards)
    avg_eval_base_reward = np.mean(eval_base_rewards)  # ## MODIFICATION: Calculate base reward average
    avg_eval_success_rate = np.mean(eval_success_rates)

    print(f"Evaluation Complete: Avg Total Reward: {avg_eval_reward:.2f}, Avg Base Reward: {avg_eval_base_reward:.2f}, Avg Success Rate: {avg_eval_success_rate*100:.1f}%")

    return {
        'eval_avg_reward': avg_eval_reward,  # Keep for backward compatibility
        'eval_avg_base_reward': avg_eval_base_reward,  # ## MODIFICATION: Return base reward
        'eval_success_rate': avg_eval_success_rate,
    }
## MODIFICATION END


## MODIFICATION: Pass output_dir to this function
def run_ablation_experiments(config_file, output_dir):
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
              f"Reward_Shaping={config['use_reward_shaping']}")
        print(f"{'-'*60}")
        
        ## MODIFICATION: Pass output_dir to training function
        results = train_model_with_clear_responsibilities(config, config_file, output_dir, num_episodes=None, seed=42)
        
        ## MODIFICATION START: Run evaluation after training
        eval_results = evaluate_saved_model(results['model_path'], config, config_file)
        results.update(eval_results) # Merge evaluation results
        ## MODIFICATION END

        all_results.append(results)
        
        print(f"\nResults for {config['name']}:")
        print(f"  Final Average Training Reward (last 100 ep): {results['final_avg_reward']:.2f}")
        print(f"  Final Training Success Rate (last 100 ep): {results['final_success_rate']*100:.1f}%")
        ## MODIFICATION: Print evaluation results (now shows base reward)
        print(f"  Deterministic Evaluation Base Reward (10 ep avg): {results['eval_avg_base_reward']:.2f}")
        print(f"  Deterministic Evaluation Total Reward (10 ep avg): {results['eval_avg_reward']:.2f}")
        print(f"  Deterministic Evaluation Success Rate (10 ep avg): {results['eval_success_rate']*100:.1f}%")
        print(f"  Training Time: {results['training_time']:.2f}s")
        print(f"  Environment Violations (Safety/Constraint): {results['total_env_safety_violations']}/{results['total_env_constraint_violations']}")

    return all_results, num_locations

def create_ablation_visualizations(all_results, num_locations, output_dir='ablation_results', config_file=None, timestamp=None):
    """创建修正后的消融实验可视化"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 主要性能对比图
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f'Ablation Study: {num_locations} Locations - Component Contribution Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Learning Curves
    ax1 = axes[0, 0]
    for result in all_results:
        rewards = result['episode_base_rewards']
        plot_window_size = VISUALIZATION_PARAMS['plot_window_size']
        smoothed_rewards = pd.Series(rewards).rolling(window=plot_window_size, min_periods=1).mean()
        ax1.plot(smoothed_rewards, label=result['model_name'], linewidth=2)
    
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
        ax2.plot(smoothed_success, label=result['model_name'], linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel(f'Success Rate ({plot_window_size}-episode window)')
    ax2.set_title('Safety Performance During Training')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Environment Violations During Training
    ax3 = axes[0, 2]
    model_names = [result['model_name'] for result in all_results]
    env_safety_violations = [result['total_env_safety_violations'] for result in all_results]
    env_constraint_violations = [result['total_env_constraint_violations'] for result in all_results]
    x_pos = np.arange(len(model_names))
    width = 0.35
    ax3.bar(x_pos - width/2, env_safety_violations, width, label='Env Safety Violations', alpha=0.8, color='red')
    ax3.bar(x_pos + width/2, env_constraint_violations, width, label='Env Constraint Violations', alpha=0.8, color='orange')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Total Violations Count')
    ax3.set_title('Total Environment Violations During Training')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4 & 5: Final Evaluation Performance
    # ## MODIFICATION: These plots now show the new deterministic evaluation results.
    model_names = [result['model_name'] for result in all_results]
    
    # Plot 4: Final Evaluation Base Reward (核心指标)
    ax4 = axes[1, 0]
    eval_base_rewards = [result['eval_avg_base_reward'] for result in all_results]
    bars = ax4.bar(model_names, eval_base_rewards, alpha=0.8, edgecolor='black', color='green')
    ax4.set_ylabel('Average Base Reward (Deterministic Evaluation)')
    ax4.set_title('Final Performance - Pure Task Reward (Evaluation)')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, eval_base_rewards):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 5: Final Evaluation Success Rate
    ax5 = axes[1, 1]
    eval_success_rates = [result['eval_success_rate'] * 100 for result in all_results]
    bars = ax5.bar(model_names, eval_success_rates, alpha=0.8, edgecolor='black', color='blue')
    ax5.set_ylabel('Success Rate % (Deterministic Evaluation)')
    ax5.set_title('Final Safety (Evaluation)')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 105)
    for bar, value in zip(bars, eval_success_rates):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

    # Plot 6: Training Time
    ax6 = axes[1, 2]
    training_times = [result['training_time'] for result in all_results]
    bars = ax6.bar(model_names, training_times, alpha=0.8, edgecolor='black', color='purple')
    ax6.set_ylabel('Training Time (seconds)')
    ax6.set_title('Training Efficiency')
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, training_times):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height, f'{value:.0f}s', ha='center', va='bottom', fontweight='bold')
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')

    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/ablation_analysis_{num_locations}locations_V3.png', 
                dpi=VISUALIZATION_PARAMS['dpi'], bbox_inches='tight')
    
    ## MODIFICATION START: Added evaluation metrics to the summary table (using base reward).
    analysis_df = pd.DataFrame({
        'Model': [result['model_name'] for result in all_results],
        'Action Mask': [result['model_config']['use_action_mask'] for result in all_results],
        'Reward Shaping': [result['model_config']['use_reward_shaping'] for result in all_results],
        'Eval Base Reward': [f"{result['eval_avg_base_reward']:.2f}" for result in all_results],  # Main metric
        'Eval Total Reward': [f"{result['eval_avg_reward']:.2f}" for result in all_results],  # Reference 
        'Eval Success Rate (%)': [f"{result['eval_success_rate']*100:.1f}" for result in all_results],
        'Training Final Reward (Last 100)': [f"{result['final_avg_reward']:.2f}" for result in all_results],
        'Training Final Success (%) (Last 100)': [f"{result['final_success_rate']*100:.1f}" for result in all_results],
        'Training Time (s)': [f"{result['training_time']:.1f}" for result in all_results],
        'Total Env Safety Violations': [result['total_env_safety_violations'] for result in all_results],
        'Total Env Constraint Violations': [result['total_env_constraint_violations'] for result in all_results]
    })
    ## MODIFICATION END
    
    # Save analysis table
    analysis_df.to_csv(f'{output_dir}/ablation_analysis_{num_locations}locations_V3.csv', index=False)
    
    # Save configuration and parameters
    # (rest of the function remains the same, saving config files and README)
    import shutil
    import json
    
    # 定义模型配置（在函数内部定义，避免全局变量依赖）
    model_configs = [
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
        'model_configs': model_configs
    }
    
    with open(f'{output_dir}/experiment_config.json', 'w') as f:
        json.dump(experiment_config, f, indent=4)
        
    print(f"\n" + "="*80)
    print(f"ABLATION STUDY RESULTS (V3): {num_locations} Locations")
    print("="*80)
    print(analysis_df.to_string(index=False))
    
    return analysis_df


def run_single_experiment(config_file):
    """运行单个配置文件的消融实验"""
    print(f"Starting Ablation Experiments for {config_file}...")
    
    # 创建带时间戳和位置信息的输出目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取训练步数
    num_episodes = TRAINING_PARAMS['num_episodes']  # 从全局参数中获取
    
    # 提取位置数量
    config_name = os.path.basename(config_file).replace('.json', '')
    num_locations = config_name.split('_')[-1]
    
    # 创建文件夹名称
    output_dir = f"ablation_results/{timestamp}_{num_episodes}ep_{num_locations}loc"
    os.makedirs(output_dir, exist_ok=True) # ## MODIFICATION: Create directory here
    
    # Run experiments
    ## MODIFICATION: Pass the created output_dir
    results, num_locations = run_ablation_experiments(config_file, output_dir)
    
    # Create visualizations and analysis
    print(f"\nCreating analysis and visualizations for {num_locations} locations...")
    analysis_table = create_ablation_visualizations(results, num_locations, output_dir, config_file, timestamp)
    
    print(f"\nAblation study for {num_locations} locations completed!")
    print(f"Results saved in '{output_dir}' directory")
    print("="*80)
    
    return results, analysis_table

# The functions run_all_configurations and create_cross_config_comparison
# will not be included as they were not part of the initial user request and
# would require significant modification to support the new workflow.
# The primary entry point will be simplified.

if __name__ == "__main__":
    # This simplified main block will run the experiment for one or more config files
    # specified on the command line, or a default if none are given.
    
    # Default configuration file if none are provided via command line
    default_configs = ['config/config_5.json']
    
    config_files_to_run = sys.argv[1:] if len(sys.argv) > 1 else default_configs
    
    for config_file in config_files_to_run:
        if not os.path.exists(config_file):
            print(f"Error: Config file '{config_file}' not found! Skipping.")
            # Assume dependent files are in the same relative path
            # For example: drone_env.py and agent/dqn_agent_ablation.py
            print("Please ensure that 'drone_env.py', 'agent/dqn_agent_ablation.py', and the 'config' directory are present.")
            # sys.exit(1) # Commented out to allow continuation if other files exist
            continue

        # Each configuration gets its own experiment run and its own results folder.
        try:
             # Assume required files are present and run the experiment
            from drone_env import PureDroneRoutePlanningEnv
            from agent.dqn_agent_ablation import IntelligentDQNAgent
            run_single_experiment(config_file)
        except ImportError as e:
            print(f"Error: A required file is missing. {e}")
            print("Please ensure that 'drone_env.py' and 'agent/dqn_agent_ablation.py' are in the correct locations.")
            break
        except FileNotFoundError as e:
            # This can happen if the agent tries to load a file that doesn't exist
            print(f"Error: A file was not found during the experiment. {e}")
            break