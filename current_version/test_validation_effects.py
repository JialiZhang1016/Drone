# test_validation_effects.py
# 测试严格动作验证对不同模型的影响

import json
import numpy as np
import time
from collections import deque
import random

from pure_drone_env import PureDroneRoutePlanningEnv
from agent.intelligent_dqn_agent import IntelligentDQNAgent

def run_quick_test(model_config, num_episodes=50):
    """运行快速测试验证环境修改效果"""
    
    print(f"测试模型: {model_config['name']}")
    
    # Load environment configuration
    with open('config/realword_8/config_8.json', 'r') as f:
        env_config = json.load(f)
    
    # 创建环境和Agent
    env = PureDroneRoutePlanningEnv(env_config)
    agent = IntelligentDQNAgent(
        env,
        use_action_mask=model_config['use_action_mask'],
        use_safety_mechanism=model_config['use_safety_mechanism'],
        use_constraint_check=model_config['use_constraint_check']
    )
    
    # 统计数据
    episode_rewards = []
    violation_episodes = 0
    violation_reasons = {}
    total_steps = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.reset_statistics()
        
        episode_reward = 0
        episode_violated = False
        steps = 0
        
        while not env.done and steps < 20:  # 限制最大步数
            # Agent选择动作
            action_idx = agent.select_action(state, epsilon=0.1)
            
            if action_idx is None:
                break
                
            action = agent.action_index_mapping[action_idx]
            
            # 环境执行动作
            next_state, reward, done, info = env.step(action)
            
            # 检查是否违规
            if info.get('violation', False):
                episode_violated = True
                violation_reason = info.get('violation_reason', 'Unknown')
                if violation_reason in violation_reasons:
                    violation_reasons[violation_reason] += 1
                else:
                    violation_reasons[violation_reason] = 1
                break
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        if episode_violated:
            violation_episodes += 1
        total_steps += steps
    
    # 计算统计信息
    avg_reward = np.mean(episode_rewards)
    violation_rate = violation_episodes / num_episodes
    avg_steps = total_steps / num_episodes
    
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  违规率: {violation_rate*100:.1f}% ({violation_episodes}/{num_episodes})")
    print(f"  平均步数: {avg_steps:.1f}")
    
    if violation_reasons:
        print(f"  违规原因统计:")
        for reason, count in violation_reasons.items():
            print(f"    {reason}: {count}次")
    
    return {
        'model_name': model_config['name'],
        'avg_reward': avg_reward,
        'violation_rate': violation_rate,
        'violation_episodes': violation_episodes,
        'violation_reasons': violation_reasons,
        'avg_steps': avg_steps
    }

def main():
    """测试不同模型配置在严格验证环境下的表现"""
    
    print("="*60)
    print("测试严格动作验证对不同模型的影响")
    print("="*60)
    
    # 测试模型配置
    model_configs = [
        {
            'name': 'Model 1: Vanilla DQN (No Intelligence)',
            'use_action_mask': False,
            'use_safety_mechanism': False,
            'use_constraint_check': False
        },
        {
            'name': 'Model 2: DQN + Action Masking',
            'use_action_mask': True,
            'use_safety_mechanism': False,
            'use_constraint_check': False
        },
        {
            'name': 'Model 5: Complete Intelligent Agent',
            'use_action_mask': True,
            'use_safety_mechanism': True,
            'use_constraint_check': True
        }
    ]
    
    results = []
    
    for config in model_configs:
        print(f"\n{'-'*40}")
        try:
            result = run_quick_test(config, num_episodes=30)
            results.append(result)
        except Exception as e:
            print(f"  错误: {e}")
            continue
        print(f"{'-'*40}")
    
    # 汇总对比
    print(f"\n" + "="*60)
    print("严格验证环境下的模型对比汇总")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  平均奖励: {result['avg_reward']:.2f}")
        print(f"  违规率: {result['violation_rate']*100:.1f}%")
        print(f"  平均步数: {result['avg_steps']:.1f}")
    
    print(f"\n关键发现:")
    print(f"1. 环境现在会严格拒绝无效动作")
    print(f"2. 无智能组件的模型(Model 1)应该有最高的违规率")
    print(f"3. 有Action Masking的模型应该违规率更低")
    print(f"4. 完整智能Agent应该违规率最低")
    
    return results

if __name__ == "__main__":
    main()