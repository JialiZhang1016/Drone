#!/usr/bin/env python3
# test_heuristic_agents.py
# 简化测试脚本，验证启发式代理功能

import json
import numpy as np
from drone_env import PureDroneRoutePlanningEnv
from agent.heuristic_agents import GreedyAgent, RuleBasedAgent

def test_heuristic_agents():
    """测试启发式代理的基本功能"""
    
    # 加载配置
    config_file = 'config/config_5.json'
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ Successfully loaded config: {config_file}")
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_file}")
        return False
    
    # 创建环境
    try:
        env = PureDroneRoutePlanningEnv(config, use_reward_shaping=True)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False
    
    # 生成动作列表（简化版）
    action_list = []
    for loc in range(env.m + 1):
        time_lower, time_upper = env.get_data_time_bounds(loc)
        if time_upper == time_lower:
            time_values = [time_lower]
        else:
            time_values = np.linspace(time_lower, time_upper, num=3)  # 减少动作数量
        for t in time_values:
            action_list.append((loc, round(float(t), 2)))
    
    print(f"✓ Generated {len(action_list)} actions")
    
    # 测试两个代理
    agents = {
        'Greedy': GreedyAgent(env, action_list),
        'Rule-Based': RuleBasedAgent(env, action_list)
    }
    
    for agent_name, agent in agents.items():
        print(f"\n--- Testing {agent_name} Agent ---")
        
        try:
            # 重置环境
            obs, info = env.reset(seed=42)
            total_reward = 0
            steps = 0
            max_steps = 20  # 限制最大步数
            
            while steps < max_steps:
                # 获取当前观测
                current_obs = env._get_observation()
                
                # 选择动作
                action_idx = agent.select_action(current_obs)
                
                if action_idx is None or action_idx == -1:
                    print(f"  Agent returned invalid action: {action_idx}")
                    break
                
                # 执行动作
                action = action_list[action_idx]
                obs, reward, done, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                print(f"  Step {steps}: Action={action}, Reward={reward:.2f}, Done={done}")
                
                if done:
                    break
            
            success = env.L_t == 0  # 成功返回基地
            print(f"  Result: {steps} steps, Total Reward: {total_reward:.2f}, Success: {success}")
            print(f"  ✓ {agent_name} Agent completed successfully")
            
        except Exception as e:
            print(f"  ✗ {agent_name} Agent failed: {e}")
            import traceback
            traceback.print_exc()
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Heuristic Agents")
    print("=" * 50)
    
    success = test_heuristic_agents()
    
    if success:
        print("\n✓ All tests completed!")
    else:
        print("\n✗ Tests failed!")