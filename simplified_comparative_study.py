#!/usr/bin/env python3
# simplified_comparative_study.py
# 不依赖Stable Baselines3的简化对比实验

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from collections import deque

from drone_env import PureDroneRoutePlanningEnv
from agent.heuristic_agents import GreedyAgent, RuleBasedAgent

class RandomAgent:
    """随机代理作为基线"""
    def __init__(self, env, action_list):
        self.env = env
        self.action_list = action_list
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}

    def select_action(self, state):
        # 过滤掉无效动作
        valid_actions = []
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        
        for idx, action in self.action_index_mapping.items():
            loc_next, t_data = action
            
            # 不能重复访问（除了基地）
            if loc_next != 0 and self.env.is_location_visited(loc_next):
                continue
                
            # 计算是否有足够时间
            t_flight = self.env.get_expected_flight_time(current_location, loc_next)
            t_return = self.env.get_expected_flight_time(loc_next, 0)
            
            if remaining_time >= t_flight + t_data + t_return:
                valid_actions.append(idx)
        
        # 如果没有有效动作，返回基地
        if not valid_actions:
            for idx, action in self.action_index_mapping.items():
                if action[0] == 0:
                    return idx
        
        return np.random.choice(valid_actions)

def generate_action_list(env, num_time_bins=3):
    """生成动作列表"""
    action_list = []
    for loc in range(env.m + 1):
        time_lower, time_upper = env.get_data_time_bounds(loc)
        if time_upper == time_lower:
            time_values = [time_lower]
        else:
            time_values = np.linspace(time_lower, time_upper, num=num_time_bins)
        for t in time_values:
            action_list.append((loc, round(float(t), 2)))
    return action_list

def evaluate_agent(agent, env, action_list, n_episodes=20, agent_name="Unknown"):
    """评估单个代理"""
    print(f"  Evaluating {agent_name}...")
    
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        total_reward = 0
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            try:
                current_obs = env._get_observation()
                action_idx = agent.select_action(current_obs)
                
                if action_idx is None or action_idx == -1:
                    break
                    
                action = action_list[action_idx]
                obs, reward, done, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
                    
            except Exception as e:
                print(f"    Error in episode {episode}: {e}")
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if env.L_t == 0 else 0)
    
    return {
        'reward_mean': np.mean(episode_rewards),
        'reward_std': np.std(episode_rewards),
        'success_rate': np.mean(episode_successes),
        'avg_steps': np.mean(episode_lengths),
        'episodes_completed': len(episode_rewards)
    }

def run_simplified_comparison(config_file, output_dir):
    """运行简化对比实验"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading config: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 创建环境
    env = PureDroneRoutePlanningEnv(config, use_reward_shaping=True)
    action_list = generate_action_list(env)
    
    print(f"Environment: {env.m} locations, {len(action_list)} actions")
    
    # 创建代理
    agents = {
        'Random': RandomAgent(env, action_list),
        'Greedy': GreedyAgent(env, action_list),
        'Rule-Based': RuleBasedAgent(env, action_list)
    }
    
    # 评估所有代理
    results = []
    for agent_name, agent in agents.items():
        result = evaluate_agent(agent, env, action_list, n_episodes=30, agent_name=agent_name)
        result['agent'] = agent_name
        results.append(result)
        print(f"  {agent_name}: Reward={result['reward_mean']:.2f}±{result['reward_std']:.2f}, "
              f"Success={result['success_rate']:.2%}, Steps={result['avg_steps']:.1f}")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)
    
    # 绘制结果
    plot_comparison_results(df, output_dir)
    
    return df

def plot_comparison_results(df, output_dir):
    """绘制对比结果"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Heuristic Agents Comparison', fontsize=16)
    
    # 平均奖励
    axes[0].bar(df['agent'], df['reward_mean'], yerr=df['reward_std'], capsize=5)
    axes[0].set_title('Average Cumulative Reward')
    axes[0].set_ylabel('Reward')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 成功率
    axes[1].bar(df['agent'], df['success_rate'])
    axes[1].set_title('Success Rate')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    # 平均步数
    axes[2].bar(df['agent'], df['avg_steps'])
    axes[2].set_title('Average Episode Length')
    axes[2].set_ylabel('Steps')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heuristic_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved to {output_dir}/heuristic_comparison.png")

if __name__ == "__main__":
    config_file = 'config/config_5.json'
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f'simplified_results/{timestamp}_heuristic_comparison'
    
    print("=" * 60)
    print("Simplified Comparative Study: Heuristic Agents")
    print("=" * 60)
    
    results_df = run_simplified_comparison(config_file, output_dir)
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("\n--- Final Summary ---")
    print(results_df.to_string(index=False))