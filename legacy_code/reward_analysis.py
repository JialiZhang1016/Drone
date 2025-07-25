# reward_analysis.py
# 分析奖励计算逻辑，验证高违规高奖励的原因

import json
import numpy as np
import matplotlib.pyplot as plt
from pure_drone_env import PureDroneRoutePlanningEnv
from agent.intelligent_dqn_agent import IntelligentDQNAgent

def analyze_vanilla_dqn_episode():
    """分析Vanilla DQN的一个典型episode，追踪奖励和违规"""
    
    # Load environment configuration
    with open('config/realword_8/config_8.json', 'r') as f:
        env_config = json.load(f)
    
    # 创建环境和Agent
    env = PureDroneRoutePlanningEnv(env_config)
    agent = IntelligentDQNAgent(
        env,
        use_action_mask=False,
        use_safety_mechanism=False,
        use_constraint_check=False
    )
    
    # 运行一个episode，详细记录
    state, _ = env.reset()
    episode_data = []
    step = 0
    
    print("=== Vanilla DQN Episode Analysis ===")
    print(f"Initial state: Location={state['current_location']}, Time={state['remaining_time'][0]:.1f}")
    print()
    
    while not env.done and step < 20:  # 限制步数防止无限循环
        # Agent选择动作
        action_idx = agent.select_action(state, epsilon=0.1)  # 低epsilon以观察学习后的策略
        
        if action_idx is None:
            print("No valid actions, episode terminated")
            break
            
        action = agent.action_index_mapping[action_idx]
        
        # 记录执行前的状态
        pre_state = {
            'step': step,
            'location': state['current_location'],
            'remaining_time': state['remaining_time'][0],
            'action': action,
            'env_safety_violations': env.safety_violations,
            'env_constraint_violations': env.constraint_violations
        }
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 记录执行后的结果
        post_state = {
            'new_location': next_state['current_location'],
            'new_remaining_time': next_state['remaining_time'][0],
            'step_reward': reward,
            'cumulative_reward': env.total_reward,
            'done': done,
            'env_safety_violations': env.safety_violations,
            'env_constraint_violations': env.constraint_violations
        }
        
        # 合并并记录
        step_data = {**pre_state, **post_state}
        episode_data.append(step_data)
        
        # 打印详细信息
        print(f"Step {step}:")
        print(f"  Action: Go to {action[0]}, collect for {action[1]:.1f} time")
        print(f"  Before: Time={pre_state['remaining_time']:.1f}, Safety={pre_state['env_safety_violations']}, Constraint={pre_state['env_constraint_violations']}")
        print(f"  After:  Time={post_state['new_remaining_time']:.1f}, Safety={post_state['env_safety_violations']}, Constraint={post_state['env_constraint_violations']}")
        print(f"  Step Reward: {reward:.2f}, Cumulative: {post_state['cumulative_reward']:.2f}")
        
        # 检查是否发生违规
        if post_state['new_remaining_time'] < 0:
            print(f"  *** CONSTRAINT VIOLATION: Time went negative! ***")
        if post_state['env_safety_violations'] > pre_state['env_safety_violations']:
            print(f"  *** SAFETY VIOLATION: Failed to return home! ***")
            
        print()
        
        state = next_state
        step += 1
    
    print(f"=== Episode Summary ===")
    print(f"Total steps: {step}")
    print(f"Final reward: {env.total_reward:.2f}")
    print(f"Safety violations: {env.safety_violations}")
    print(f"Constraint violations: {env.constraint_violations}")
    print(f"Success rate: {100 if next_state['current_location'] == 0 else 0}%")
    
    # 分析奖励构成
    positive_rewards = sum([s['step_reward'] for s in episode_data if s['step_reward'] > 0])
    negative_rewards = sum([s['step_reward'] for s in episode_data if s['step_reward'] < 0])
    
    print(f"\nReward Analysis:")
    print(f"Positive rewards (data collection): {positive_rewards:.2f}")
    print(f"Negative rewards (costs + penalties): {negative_rewards:.2f}")
    print(f"Net reward: {positive_rewards + negative_rewards:.2f}")
    
    return episode_data

def compare_reward_mechanisms():
    """比较不同奖励机制的效果"""
    
    print("\n" + "="*60)
    print("REWARD MECHANISM ANALYSIS")
    print("="*60)
    
    # 分析当前奖励机制的问题
    print("\nCurrent Reward Mechanism Issues:")
    print("1. High data collection rewards (10x time for HC, 2x time for LC)")
    print("2. Low flight costs (-1x time)")
    print("3. Penalty only applied at episode end (-10000)")
    print("4. No immediate penalty for constraint violations")
    
    # 建议的改进
    print("\nSuggested Improvements:")
    print("1. Reduce data collection reward multipliers")
    print("2. Increase flight costs") 
    print("3. Add immediate penalties for violations")
    print("4. Progressive penalties for continued violations")
    
    # 示例计算
    print("\nExample Calculation (HC location, 100 time units):")
    print("Current system:")
    print("  Data reward: 10 * 100 = 1000")
    print("  Flight cost: -1 * 50 = -50 (example)")
    print("  Net per step: ~950")
    print("  After 10 such steps: ~9500")
    print("  Final penalty (if not home): -10000")
    print("  Net result: Still could be positive!")
    
    print("\nImproved system suggestion:")
    print("  Data reward: 3 * 100 = 300")
    print("  Flight cost: -2 * 50 = -100")
    print("  Violation penalty: -500 (immediate)")
    print("  Net per step: ~100 (if valid), -500 (if violation)")

if __name__ == "__main__":
    # 分析一个典型的Vanilla DQN episode
    episode_data = analyze_vanilla_dqn_episode()
    
    # 分析奖励机制
    compare_reward_mechanisms()
    
    print(f"\n{'='*60}")
    print("CONCLUSION: The high reward despite violations is due to:")
    print("1. Large positive rewards accumulated before violations")
    print("2. Delayed penalty application (only at episode end)")
    print("3. Penalty magnitude insufficient to offset accumulated rewards")
    print("="*60)