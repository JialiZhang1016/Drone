# test_environment_validation.py
# 测试修改后的环境严格动作验证功能

import json
import numpy as np
from pure_drone_env import PureDroneRoutePlanningEnv

def test_environment_validation():
    """测试环境的动作验证功能"""
    
    print("=" * 60)
    print("测试环境严格动作验证功能")
    print("=" * 60)
    
    # Load environment configuration
    with open('config/realword_8/config_8.json', 'r') as f:
        env_config = json.load(f)
    
    # 创建环境
    env = PureDroneRoutePlanningEnv(env_config)
    
    # 测试用例
    test_cases = [
        {
            'name': '正常动作测试',
            'action': (1, 100.0),  # 去位置1，收集100单位数据
            'expected_violation': False,
            'description': '从Home去位置1，正常数据收集时间'
        },
        {
            'name': '无效位置测试1',
            'action': (-1, 100.0),  # 无效位置
            'expected_violation': True,
            'description': '尝试去位置-1（无效位置）'
        },
        {
            'name': '无效位置测试2', 
            'action': (10, 100.0),  # 超出范围
            'expected_violation': True,
            'description': '尝试去位置10（超出范围，环境只有8个位置+Home）'
        },
        {
            'name': '重复访问测试',
            'setup_actions': [(1, 100.0)],  # 先访问位置1
            'action': (1, 50.0),  # 再次尝试访问位置1
            'expected_violation': True,
            'description': '尝试重复访问已访问的位置1'
        },
        {
            'name': '数据收集时间过小',
            'action': (2, 0.0),  # 数据收集时间为0
            'expected_violation': True,
            'description': '数据收集时间小于最小值'
        },
        {
            'name': '数据收集时间过大',
            'action': (3, 1000.0),  # 数据收集时间过大
            'expected_violation': True,
            'description': '数据收集时间大于最大值'
        },
        {
            'name': '原地不动测试',
            'action': (0, 0.0),  # 尝试原地不动
            'expected_violation': True,
            'description': '尝试原地不动（从Home到Home）'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['name']}")
        print(f"描述: {test_case['description']}")
        
        # 重置环境
        state, _ = env.reset()
        print(f"初始状态: 位置{state['current_location']}, 剩余时间{state['remaining_time'][0]:.1f}")
        
        # 执行预设动作（如果有）
        if 'setup_actions' in test_case:
            for setup_action in test_case['setup_actions']:
                next_state, reward, done, info = env.step(setup_action)
                print(f"预设动作: 去位置{setup_action[0]}, 收集{setup_action[1]}单位数据")
                print(f"  -> 现在位置{next_state['current_location']}, 剩余时间{next_state['remaining_time'][0]:.1f}")
                if done:
                    break
        
        # 执行测试动作
        test_action = test_case['action']
        print(f"测试动作: 去位置{test_action[0]}, 收集{test_action[1]}单位数据")
        
        try:
            next_state, reward, done, info = env.step(test_action)
            
            is_violation = info.get('violation', False)
            violation_reason = info.get('violation_reason', 'None')
            
            print(f"结果:")
            print(f"  奖励: {reward}")
            print(f"  违规: {is_violation}")
            print(f"  违规原因: {violation_reason}")
            print(f"  Episode结束: {done}")
            print(f"  最终位置: {next_state['current_location']}")
            
            # 验证结果
            if is_violation == test_case['expected_violation']:
                print(f"  ✓ 测试通过")
            else:
                print(f"  ✗ 测试失败！期望违规:{test_case['expected_violation']}, 实际违规:{is_violation}")
                
        except Exception as e:
            print(f"  ✗ 测试出错: {e}")
    
    print(f"\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

def test_valid_episode():
    """测试一个完整的有效episode"""
    
    print(f"\n" + "=" * 60)
    print("测试完整有效Episode")
    print("=" * 60)
    
    # Load environment configuration
    with open('config/realword_8/config_8.json', 'r') as f:
        env_config = json.load(f)
    
    # 创建环境
    env = PureDroneRoutePlanningEnv(env_config)
    
    # 重置环境
    state, _ = env.reset()
    print(f"初始状态: 位置{state['current_location']}, 剩余时间{state['remaining_time'][0]:.1f}")
    
    # 执行一系列有效动作
    valid_actions = [
        (1, 100.0),  # 去位置1收集数据
        (2, 150.0),  # 去位置2收集数据
        (3, 80.0),   # 去位置3收集数据
        (0, 0.0)     # 返回Home
    ]
    
    total_reward = 0
    step = 0
    
    for action in valid_actions:
        if env.done:
            break
            
        step += 1
        print(f"\nStep {step}: 去位置{action[0]}, 收集{action[1]}单位数据")
        
        try:
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            is_violation = info.get('violation', False)
            
            print(f"  奖励: {reward:.2f}")
            print(f"  总奖励: {total_reward:.2f}")
            print(f"  违规: {is_violation}")
            print(f"  当前位置: {next_state['current_location']}")
            print(f"  剩余时间: {next_state['remaining_time'][0]:.1f}")
            print(f"  Episode结束: {done}")
            
            if is_violation:
                print(f"  违规原因: {info.get('violation_reason', 'Unknown')}")
                break
                
        except Exception as e:
            print(f"  错误: {e}")
            break
    
    print(f"\nEpisode总结:")
    print(f"  总步数: {step}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  访问路径: {env.visit_order}")
    print(f"  成功完成: {not env.done or state['current_location'] == 0}")

if __name__ == "__main__":
    # 测试动作验证功能
    test_environment_validation()
    
    # 测试完整有效episode
    test_valid_episode()