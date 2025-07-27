# comparative_study.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from collections import deque

# 导入您自己的环境和智能体
from drone_env import PureDroneRoutePlanningEnv
# from agent.dqn_agent_ablation import IntelligentDQNAgent, ExperienceReplay  # 未使用，暂时注释
from agent.heuristic_agents import GreedyAgent, RuleBasedAgent

# 导入Stable Baselines3
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# ==================================
# 1. 环境封装 (用于适配Stable Baselines3)
# ==================================
class SB3DroneEnv(gym.Wrapper):
    """
    一个封装器，用于将我们的自定义环境适配给Stable Baselines3。
    主要做两件事：
    1. 将字典(Dict)观测空间扁平化为向量(Box)。
    2. 将元组(Tuple)动作空间离散化为(Discrete)。
    """
    def __init__(self, config_file, use_reward_shaping=True):
        # 首先加载原始环境
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.raw_env = PureDroneRoutePlanningEnv(config, use_reward_shaping)
        super().__init__(self.raw_env)
        
        # 1. 离散化动作空间
        self.action_list = self._discretize_actions()
        self.action_space = spaces.Discrete(len(self.action_list))
        
        # 2. 扁平化观测空间
        self.observation_space = self._flatten_obs_space()

    def _discretize_actions(self, num_time_bins=5):
        action_list = []
        for loc in range(self.raw_env.m + 1):
            time_lower, time_upper = self.raw_env.get_data_time_bounds(loc)
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=num_time_bins)
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
        
    def _flatten_obs_space(self):
        # 原始观测空间: Dict('current_location', 'remaining_time', 'visited', 'weather')
        # 扁平化后: Box(shape=(1 + 1 + (m+1) + 3,))
        m = self.raw_env.m
        flat_space_size = 1 + 1 + (m + 1) + 3 # location, time, visited, weather_one_hot
        return spaces.Box(low=-np.inf, high=np.inf, shape=(flat_space_size,), dtype=np.float32)

    def _get_flat_obs(self, obs_dict):
        loc = obs_dict['current_location'] / (self.raw_env.m + 1) # 归一化
        time_rem = obs_dict['remaining_time'][0] / self.raw_env.T_max # 归一化
        visited = obs_dict['visited']
        weather = obs_dict['weather']
        weather_one_hot = np.zeros(3)
        weather_one_hot[weather] = 1
        
        return np.concatenate(([loc], [time_rem], visited, weather_one_hot)).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs_dict, info = self.raw_env.reset(seed=seed)
        return self._get_flat_obs(obs_dict), info

    def step(self, action):
        # 将离散动作映射回元组动作
        tuple_action = self.action_list[action]
        obs_dict, reward, done, info = self.raw_env.step(tuple_action)
        
        # 旧版Gym API返回4个值，新版返回5个
        # 我们需要兼容SB3，它期望5个返回值
        truncated = info.get('TimeLimit.truncated', False)

        return self._get_flat_obs(obs_dict), reward, done, truncated, info

# ========================================
# 2. 实验评估与运行逻辑
# ========================================

def evaluate_policy(model, env, n_eval_episodes=100, is_heuristic=False):
    """评估一个策略 (RL或启发式)"""
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_safety_violations = []
    episode_constraint_violations = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        length = 0
        while not done:
            if is_heuristic:
                # 启发式智能体需要原始的字典格式观测
                raw_obs = env.unwrapped._get_observation()
                action = model.select_action(raw_obs)
            else: # SB3 RL Agent
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            total_reward += reward
            length += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        # 成功定义为: 任务结束且在基地
        is_success = (env.unwrapped.L_t == 0)
        episode_successes.append(1 if is_success else 0)
        episode_safety_violations.append(1 if not is_success else 0)
        episode_constraint_violations.append(env.unwrapped.constraint_violations)
    
    return {
        'reward_mean': np.mean(episode_rewards),
        'reward_std': np.std(episode_rewards),
        'success_rate_mean': np.mean(episode_successes),
        'safety_violations_mean': np.mean(episode_safety_violations),
        'constraint_violations_mean': np.mean(episode_constraint_violations),
    }


def run_experiments(config_file, output_dir):
    """运行所有对比实验"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 实验配置 ---
    N_SEEDS = 3  # 为节省时间，使用3个随机种子，实际研究建议5-10个
    TRAIN_TIMESTEPS = 30000 # 训练步数
    EVAL_EPISODES = 50 # 评估回合数
    
    models_to_run = {
        'PPO': {'class': PPO, 'policy': 'MlpPolicy'},
        'A2C': {'class': A2C, 'policy': 'MlpPolicy'},
        'Greedy': {'class': GreedyAgent, 'is_heuristic': True},
        'Rule-Based': {'class': RuleBasedAgent, 'is_heuristic': True},
    }
    
    results = []

    for model_name, model_info in models_to_run.items():
        print(f"\n===== Running: {model_name} on {config_file} =====")
        for seed in range(N_SEEDS):
            print(f"  - Seed: {seed+1}/{N_SEEDS}")
            start_time = time.time()
            
            # 创建环境
            env = SB3DroneEnv(config_file)
            env.reset(seed=seed)
            env.action_space.seed(seed)

            # 训练或初始化模型
            if model_info.get('is_heuristic', False):
                model = model_info['class'](env.unwrapped, env.action_list)
                train_time = 0
            else: # RL Agents
                # 使用Monitor来记录训练过程中的奖励
                log_dir = os.path.join(output_dir, "sb3_logs/")
                env = Monitor(env, log_dir)
                
                model = model_info['class'](model_info['policy'], env, verbose=0, seed=seed)
                model.learn(total_timesteps=TRAIN_TIMESTEPS)
                train_time = time.time() - start_time

            # 评估模型
            eval_results = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES, is_heuristic=model_info.get('is_heuristic', False))
            
            # 记录结果
            res = {
                'model': model_name,
                'seed': seed,
                'train_time': train_time,
                **eval_results # 合并评估结果字典
            }
            results.append(res)
    
    return pd.DataFrame(results)

# ==================================
# 3. 结果可视化
# ==================================

def plot_results(df, output_dir):
    """生成并保存结果图表"""
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 1. 最终性能柱状图 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comparative Study: Final Performance', fontsize=16)

    # a) 平均奖励
    sns.barplot(data=df, x='model', y='reward_mean', ax=axes[0], ci='sd', capsize=.1)
    axes[0].set_title('Average Cumulative Reward')
    axes[0].set_ylabel('Reward')
    axes[0].set_xlabel('Model')
    
    # b) 成功率
    sns.barplot(data=df, x='model', y='success_rate_mean', ax=axes[1], ci='sd', capsize=.1)
    axes[1].set_title('Success Rate (Safe Return to Base)')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_xlabel('Model')
    axes[1].set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'final_performance_comparison.png'), dpi=300)
    plt.close()
    print(f"✓ Saved final performance plot to {output_dir}")


def aggregate_and_save_results(df, output_dir):
    """聚合结果并保存为CSV"""
    summary = df.groupby('model').agg({
        'reward_mean': ['mean', 'std'],
        'success_rate_mean': ['mean', 'std'],
        'safety_violations_mean': ['mean', 'std'],
        'train_time': ['mean']
    }).reset_index()
    
    # 格式化列名
    summary.columns = ['Model', 'Reward (Mean)', 'Reward (Std)', 
                       'Success Rate (Mean)', 'Success Rate (Std)',
                       'Safety Violations (Mean)', 'Safety Violations (Std)',
                       'Training Time (s)']
    
    summary.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    print(f"✓ Saved aggregated results to {output_dir}/results_summary.csv")
    print("\n--- Results Summary ---")
    print(summary.to_string(index=False))


# ==================================
# 4. 主函数
# ==================================

if __name__ == "__main__":
    # --- 选择一个配置进行实验 ---
    # 您可以修改这里来测试不同的环境配置
    CONFIG_FILE = 'config/config_10.json'
    
    # 创建带时间戳的输出目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    OUTPUT_DIR = f'comparative_results/{timestamp}_{os.path.basename(CONFIG_FILE).replace(".json", "")}'
    
    print("="*50)
    print(f"Starting Comparative Study")
    print(f"Config: {CONFIG_FILE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*50)
    
    # 运行实验
    results_df = run_experiments(CONFIG_FILE, OUTPUT_DIR)
    
    # 可视化与保存结果
    plot_results(results_df, OUTPUT_DIR)
    aggregate_and_save_results(results_df, OUTPUT_DIR)
    
    print("\n✓ All experiments completed successfully!")