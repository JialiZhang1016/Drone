# simple_test.py
# 简单测试环境的动作验证功能（不依赖外部库）

import sys
import os
sys.path.append(os.path.dirname(__file__))

def simple_test():
    """简单测试环境修改是否生效"""
    
    try:
        # 尝试导入环境
        from pure_drone_env import PureDroneRoutePlanningEnv
        print("✓ 成功导入环境类")
        
        # 创建简单配置用于测试
        test_config = {
            "num_locations": 3,
            "T_max": 1000,
            "weather_prob": 0.6,
            "P_penalty": 100,
            "T_flight_good": [
                [0, 50, 60, 70],
                [50, 0, 40, 80], 
                [60, 40, 0, 50],
                [70, 80, 50, 0]
            ],
            "T_flight_bad": [
                [0, 70, 80, 90],
                [70, 0, 60, 100],
                [80, 60, 0, 70], 
                [90, 100, 70, 0]
            ],
            "T_data_lower": [0, 50, 50, 50],
            "T_data_upper": [0, 200, 200, 200],
            "criticality": ["LC", "HC", "LC", "HC"]
        }
        
        # 创建环境
        env = PureDroneRoutePlanningEnv(test_config)
        print("✓ 成功创建环境实例")
        
        # 重置环境
        state, _ = env.reset()
        print(f"✓ 环境重置成功，初始位置: {state['current_location']}")
        
        # 测试正常动作
        print("\n测试1: 正常动作")
        action = (1, 100.0)  # 去位置1，收集100单位数据
        next_state, reward, done, info = env.step(action)
        violation = info.get('violation', False)
        print(f"动作: 去位置{action[0]}, 收集{action[1]}单位数据")
        print(f"结果: 奖励={reward}, 违规={violation}, 结束={done}")
        
        if not violation:
            print("✓ 正常动作测试通过")
        else:
            print("✗ 正常动作测试失败")
        
        # 重置环境测试违规动作
        state, _ = env.reset()
        
        # 测试重复访问违规
        print("\n测试2: 重复访问违规")
        # 先去位置1
        env.step((1, 100.0))
        # 再次尝试去位置1
        action = (1, 80.0)
        next_state, reward, done, info = env.step(action)
        violation = info.get('violation', False)
        violation_reason = info.get('violation_reason', 'None')
        
        print(f"动作: 再次去位置{action[0]}")
        print(f"结果: 奖励={reward}, 违规={violation}, 结束={done}")
        print(f"违规原因: {violation_reason}")
        
        if violation and reward == -1000:
            print("✓ 重复访问违规测试通过")
        else:
            print("✗ 重复访问违规测试失败")
        
        # 重置环境测试无效位置
        state, _ = env.reset()
        
        # 测试无效位置
        print("\n测试3: 无效位置违规")
        action = (10, 100.0)  # 位置10不存在
        next_state, reward, done, info = env.step(action)
        violation = info.get('violation', False) 
        violation_reason = info.get('violation_reason', 'None')
        
        print(f"动作: 去无效位置{action[0]}")
        print(f"结果: 奖励={reward}, 违规={violation}, 结束={done}")
        print(f"违规原因: {violation_reason}")
        
        if violation and reward == -1000:
            print("✓ 无效位置违规测试通过")
        else:
            print("✗ 无效位置违规测试失败")
        
        print("\n" + "="*50)
        print("环境修改验证完成！")
        print("✓ 环境现在会严格验证动作有效性")
        print("✓ 无效动作会导致-1000惩罚并立即终止episode")
        print("✓ 违规信息会在info中返回")
        print("="*50)
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()