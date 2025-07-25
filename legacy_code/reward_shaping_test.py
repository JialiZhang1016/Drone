# reward_shaping_test.py
# 测试新的奖励塑造机制效果

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pure_drone_env import PureDroneRoutePlanningEnv
from agent.intelligent_dqn_agent import IntelligentDQNAgent

def test_reward_shaping_mechanism():
    """测试奖励塑造机制在不同情况下的表现"""
    
    # Load environment configuration
    with open('config/realword_8/config_8.json', 'r') as f:
        env_config = json.load(f)
    
    # 创建环境
    env = PureDroneRoutePlanningEnv(env_config)
    
    print("=== 奖励塑造机制测试 ===")
    print(f"初始配置：")
    print(f"  最大飞行时间: {env.T_max}")
    print(f"  惩罚系数: {env.P_penalty}")
    print()
    
    # 测试不同危险场景
    test_scenarios = [
        {
            'name': '安全场景',
            'remaining_time': 3000,
            'current_location': 1,
            'description': '充足时间，低风险'
        },
        {
            'name': '中等风险场景', 
            'remaining_time': 1000,
            'current_location': 5,
            'description': '时间适中，需要考虑返回'
        },
        {
            'name': '高风险场景',
            'remaining_time': 200,
            'current_location': 7,
            'description': '时间紧张，高风险'
        },
        {
            'name': '极度危险场景',
            'remaining_time': 50,
            'current_location': 8,
            'description': '时间极少，极高风险'
        },
        {
            'name': '时间耗尽场景',
            'remaining_time': -100,
            'current_location': 6,
            'description': '时间已用尽'
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        # 设置环境状态
        env.reset()
        env.T_t_rem = scenario['remaining_time']
        env.L_t = scenario['current_location']
        
        # 获取危险信息
        danger_info = env._get_danger_info()
        
        # 模拟一个数据收集动作（HC地点，收集100单位时间）
        test_action_location = 5  # HC地点
        test_action_time = 100
        test_flight_time = 50  # 假设飞行时间
        
        # 计算旧奖励机制的奖励（用于对比）
        old_reward = 10 * test_action_time - 1 * (test_action_time + test_flight_time)
        
        # 计算新奖励机制的奖励
        # 暂时设置环境状态以计算奖励
        old_location = env.L_t
        env.L_t = test_action_location
        new_reward = env._calculate_reward(test_action_location, test_action_time, test_flight_time)
        env.L_t = old_location  # 恢复状态
        
        result = {
            'scenario': scenario['name'],
            'remaining_time': scenario['remaining_time'],
            'location': scenario['current_location'],
            'return_time_needed': danger_info['return_time_needed'],
            'danger_ratio': danger_info['danger_ratio'],
            'danger_level': danger_info['danger_level'],
            'danger_penalty': danger_info['danger_penalty'],
            'old_reward': old_reward,
            'new_reward': new_reward,
            'reward_change': new_reward - old_reward
        }
        
        results.append(result)
        
        print(f"场景: {scenario['name']}")
        print(f"  描述: {scenario['description']}")
        print(f"  剩余时间: {scenario['remaining_time']:.1f}")
        print(f"  当前位置: {scenario['current_location']}")
        print(f"  返回所需时间: {danger_info['return_time_needed']:.1f}")
        print(f"  危险比例: {danger_info['danger_ratio']:.2f}")
        print(f"  危险等级: {danger_info['danger_level']}")
        print(f"  危险惩罚: {danger_info['danger_penalty']:.1f}")
        print(f"  旧奖励机制: {old_reward:.1f}")
        print(f"  新奖励机制: {new_reward:.1f}")
        print(f"  奖励变化: {new_reward - old_reward:.1f}")
        print()
    
    return results

def compare_vanilla_dqn_performance():
    """比较Vanilla DQN在新旧奖励机制下的表现"""
    
    print("=== Vanilla DQN 新旧奖励机制对比 ===")
    
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
    
    # 运行几个episode测试
    episodes_to_test = 5
    episode_results = []
    
    for episode in range(episodes_to_test):
        state, _ = env.reset()
        
        episode_data = {
            'episode': episode,
            'steps': 0,
            'total_reward': 0,
            'danger_penalties': 0,
            'safety_violations': 0,
            'constraint_violations': 0,
            'success': False,
            'step_details': []
        }
        
        step = 0
        while not env.done and step < 20:
            # Agent选择动作
            action_idx = agent.select_action(state, epsilon=0.1)
            
            if action_idx is None:
                break
                
            action = agent.action_index_mapping[action_idx]
            
            # 记录危险信息
            danger_info = env._get_danger_info()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            step_detail = {
                'step': step,
                'action': action,
                'danger_level': danger_info['danger_level'],
                'danger_penalty': danger_info['danger_penalty'],
                'step_reward': reward,
                'remaining_time': next_state['remaining_time'][0]
            }
            
            episode_data['step_details'].append(step_detail)
            episode_data['danger_penalties'] += danger_info['danger_penalty']
            
            state = next_state
            step += 1
        
        episode_data['steps'] = step
        episode_data['total_reward'] = env.total_reward
        episode_data['safety_violations'] = env.safety_violations
        episode_data['constraint_violations'] = env.constraint_violations
        episode_data['success'] = (state['current_location'] == 0)
        
        episode_results.append(episode_data)
        
        print(f"Episode {episode}:")
        print(f"  步数: {episode_data['steps']}")
        print(f"  总奖励: {episode_data['total_reward']:.2f}")
        print(f"  危险惩罚总计: {episode_data['danger_penalties']:.2f}")
        print(f"  安全违规: {episode_data['safety_violations']}")
        print(f"  约束违规: {episode_data['constraint_violations']}")
        print(f"  成功返回: {'是' if episode_data['success'] else '否'}")
        
        # 显示危险等级分布
        danger_levels = [step['danger_level'] for step in episode_data['step_details']]
        danger_counts = {}
        for level in danger_levels:
            danger_counts[level] = danger_counts.get(level, 0) + 1
        print(f"  危险等级分布: {danger_counts}")
        print()
    
    # 汇总统计
    avg_reward = np.mean([ep['total_reward'] for ep in episode_results])
    avg_danger_penalty = np.mean([ep['danger_penalties'] for ep in episode_results])
    success_rate = np.mean([ep['success'] for ep in episode_results])
    total_violations = sum([ep['safety_violations'] + ep['constraint_violations'] 
                           for ep in episode_results])
    
    print(f"=== 汇总统计 ===")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均危险惩罚: {avg_danger_penalty:.2f}")
    print(f"成功率: {success_rate*100:.1f}%")
    print(f"总违规次数: {total_violations}")
    
    return episode_results

def visualize_reward_shaping_effects(test_results):
    """Visualize the effects of reward shaping"""
    
    # Create DataFrame
    df = pd.DataFrame(test_results)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reward Shaping Mechanism Analysis', fontsize=16, fontweight='bold')
    
    # 1. Risk Level vs Reward Change
    ax1 = axes[0, 0]
    scenarios = df['scenario']
    reward_changes = df['reward_change']
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    bars = ax1.bar(scenarios, reward_changes, color=colors, alpha=0.7)
    ax1.set_ylabel('Reward Change')
    ax1.set_title('Reward Adjustment Under Different Risk Scenarios')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, reward_changes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 2. Danger Ratio vs Danger Penalty
    ax2 = axes[0, 1]
    danger_ratios = [min(r, 3) for r in df['danger_ratio']]  # Limit display range
    danger_penalties = df['danger_penalty']
    
    ax2.scatter(danger_ratios, danger_penalties, c=colors, s=100, alpha=0.7)
    ax2.set_xlabel('Danger Ratio')
    ax2.set_ylabel('Danger Penalty')
    ax2.set_title('Danger Ratio vs Penalty Intensity')
    ax2.grid(True, alpha=0.3)
    
    # 3. Old vs New Reward Mechanism Comparison
    ax3 = axes[1, 0]
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    ax3.bar(x_pos - width/2, df['old_reward'], width, label='Old Mechanism', alpha=0.7)
    ax3.bar(x_pos + width/2, df['new_reward'], width, label='New Mechanism', alpha=0.7)
    
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Reward Value')
    ax3.set_title('Old vs New Reward Mechanism Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenarios, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Danger Level Distribution
    ax4 = axes[1, 1]
    danger_levels = df['danger_level']
    level_counts = danger_levels.value_counts()
    
    ax4.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
            colors=['green', 'yellow', 'orange', 'red', 'darkred'])
    ax4.set_title('Danger Level Distribution')
    
    plt.tight_layout()
    plt.savefig('reward_shaping_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("开始测试奖励塑造机制...")
    
    # 1. 测试奖励塑造机制
    test_results = test_reward_shaping_mechanism()
    
    # 2. 比较Vanilla DQN表现
    episode_results = compare_vanilla_dqn_performance()
    
    # 3. 可视化效果
    visualize_reward_shaping_effects(test_results)
    
    print("Testing completed! New reward shaping mechanism implemented:")
    print("- Immediate penalty based on danger coefficient")
    print("- Progressive risk warning mechanism")
    print("- Balanced reward-cost ratio")
    print("- Immediate feedback instead of delayed punishment")