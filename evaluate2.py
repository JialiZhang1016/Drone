import gymnasium as gym
from drone_env import DroneRoutePlanningEnv
from agent.dqn_agent import DQNAgent
import json
import torch
import numpy as np
import random
import sys
import os
import time
import secrets

def evaluate_dqn(
    config_file: str,
    model_path: str,
    seed: int = None,
    verbose: bool = True,
    use_action_mask: bool = True
) -> dict:
    """
    Evaluate a trained DQN agent on the DroneRoutePlanningEnv for a single episode.

    Parameters:
        config_file (str): Path to the configuration JSON file.
        model_path (str): Path to the trained policy network parameters (.pth file).
        seed (int, optional): Random seed for reproducibility. If None, a random seed is generated.
        verbose (bool, optional): If True, prints actions and episode results. Defaults to True.
        use_action_mask (bool, optional): Whether to use action masking. Defaults to True.

    Returns:
        dict: A dictionary containing episode results, including 'episode', 'total_reward', 'steps', and 'seed'.
    """
    
    # Generate a random seed if none is provided
    if seed is None:
        seed = secrets.randbelow(2**32)  # 使用 secrets 生成高熵的种子
    
    if verbose:
        print(f"使用的 seed: {seed}")
    
    # 设置随机种子以确保可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 载入配置文件
    try:
        with open(config_file, 'r') as config_file_handle:
            config = json.load(config_file_handle)
    except FileNotFoundError:
        print(f"配置文件未找到: {config_file}")
        return {}
    except json.JSONDecodeError:
        print(f"配置文件 JSON 解析错误: {config_file}")
        return {}
    
    # 创建环境
    try:
        env = DroneRoutePlanningEnv(config)
    except Exception as e:
        print(f"创建环境时出错: {e}")
        return {}
    
    # 初始化代理
    try:
        agent = DQNAgent(env, use_action_mask=use_action_mask)
    except Exception as e:
        print(f"初始化 DQNAgent 时出错: {e}")
        return {}
    
    # 加载训练好的模型参数
    if not os.path.isfile(model_path):
        print(f"模型文件未找到: {model_path}")
        return {}
    
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        agent.policy_net.eval()
    except Exception as e:
        print(f"加载模型参数时出错: {e}")
        return {}
    
    # 开始评估
    try:
        state, _ = env.reset(seed=seed)
    except Exception as e:
        print(f"重置环境时出错: {e}")
        return {}
    
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action_idx = agent.select_action(state, epsilon=0.0)
            if action_idx is None:
                if verbose:
                    print(f"没有可用的动作。结束评估。")
                break  # 没有有效动作，结束评估
            action = agent.action_index_mapping[action_idx]
            if verbose:
                print(f"步骤 {steps + 1}: 执行动作: {action}")
            try:
                next_state, reward, done, info = env.step(action)
            except Exception as e:
                print(f"环境步进时出错: {e}")
                break
            total_reward += reward
            state = next_state
            steps += 1
    
    if verbose:
        print(f"评估完成, 总奖励: {total_reward:.2f}, 步数: {steps}, Seed: {seed}")
    
    # 关闭环境
    env.close()
    
    # 返回评估结果
    return {
        'total_reward': total_reward,
        'steps': steps,
        'seed': seed
    }

if __name__ == "__main__":
    # 配置文件和模型路径
    # config_file = "config/updated_configs/config_10_wp_0.5.json"
    # model_path = "outputs/2024-11-06_20:43:11_wp_0.5_10_5000/policy_net.pth"

    config_file = "config/updated_configs/config_10_wp_0.2.json"
    model_path = "outputs/2024-11-06_20:42:11_wp_0.2_10_5000/policy_net.pth"
    
    # 评估次数
    num_evaluations = 10
    evaluation_results = []
    
    for eval_num in range(1, num_evaluations + 1):
        # print(f"\n开始第 {eval_num} 次评估:")
        # 生成随机 seed
        seed = secrets.randbelow(2**32)
        
        # 调用 evaluate_dqn 进行单次评估
        result = evaluate_dqn(
            config_file=config_file,
            model_path=model_path,
            seed=seed,
            verbose=True,
            use_action_mask=False
        )
        
        if result:
            # 添加评估编号
            result['evaluation'] = eval_num
            evaluation_results.append(result)
        else:
            print(f"第 {eval_num} 次评估失败。")
    
    # 计算平均结果
    if evaluation_results:
        avg_reward = np.mean([res['total_reward'] for res in evaluation_results])
        avg_steps = np.mean([res['steps'] for res in evaluation_results])
        print(f"\n{num_evaluations} 次评估的平均奖励: {avg_reward:.2f}")
        print(f"{num_evaluations} 次评估的平均步数: {avg_steps:.2f}")
        
        # 可选：保存评估结果到文件
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=4)
            print(f"评估结果已保存到 {results_file}")
        except Exception as e:
            print(f"保存评估结果时出错: {e}")
    else:
        print("没有成功的评估结果。")
