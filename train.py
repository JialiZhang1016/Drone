import gymnasium as gym
from drone_env import DroneRoutePlanningEnv
from agent.dqn_agent import DQNAgent
import json
import torch
import random
import numpy as np
from collections import deque
import os
from datetime import datetime

# 加载配置文件
with open('config/config.json', 'r') as config_file:
    config = json.load(config_file) 

# 创建环境
env = DroneRoutePlanningEnv(config)

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)

# 初始化智能体
agent = DQNAgent(env)

# 超参数
num_episodes = 2000
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.001
epsilon_decay = 300
target_update_freq = 10
memory_size = 10000
gamma = 0.99

memory = deque(maxlen=memory_size)
steps_done = 0

# 获取地点数量（不包括'home'）
num_locations = config['num_locations'] # 不包括'home'

# 获取当前时间
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# 创建结果保存的目录
results_dir = f"runs/{num_locations}_{num_episodes}_{current_time}"
os.makedirs(results_dir, exist_ok=True)

# 初始化指标列表
all_rewards = []
all_losses = []
episode_lengths = []
epsilons = []
successes = []
success_rates = []
average_rewards = []
moving_average_rewards = []

# 定义统计间隔
success_rate_interval = 10  # 每10个Episode统计一次成功率
average_reward_interval = 20  # 每20个Episode统计一次平均奖励
moving_average_interval = 15  # 移动平均奖励的窗口大小

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    total_loss = 0
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
    epsilons.append(epsilon)
    steps = 0  # 记录每个Episode的步数

    while True:
        steps_done += 1
        steps += 1
        action_idx = agent.select_action(state, epsilon)
        if action_idx is None:
            break  # 无有效动作，结束Episode
        action = agent.action_index_mapping[action_idx]
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # 存储经验
        memory.append((state, action_idx, reward, next_state, done))

        state = next_state

        # 经验回放
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            loss = agent.optimize_model(batch)
            if loss is not None:
                total_loss += loss.item()

        if done:
            break

    # 更新目标网络
    if episode % target_update_freq == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    all_rewards.append(total_reward)
    episode_lengths.append(steps)
    average_loss = total_loss / steps if steps > 0 else 0
    all_losses.append(average_loss)

    # 判断是否成功（根据total_reward）
    success = total_reward >= 0
    successes.append(success)

    # 计算成功率
    if (episode + 1) % success_rate_interval == 0:
        recent_successes = successes[-success_rate_interval:]
        success_rate = sum(recent_successes) / len(recent_successes)
        success_rates.append(success_rate)
        # print(f"Episode {episode + 1}/{num_episodes}, Success Rate (last {success_rate_interval} episodes): {success_rate:.2f}")

    # 计算平均奖励
    if (episode + 1) % average_reward_interval == 0:
        recent_rewards = all_rewards[-average_reward_interval:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        average_rewards.append(avg_reward)
        #print(f"Episode {episode + 1}/{num_episodes}, Average Reward (last {average_reward_interval} episodes): {avg_reward:.2f}")

    # 计算移动平均奖励
    if len(all_rewards) >= moving_average_interval:
        recent_rewards = all_rewards[-moving_average_interval:]
        moving_avg_reward = sum(recent_rewards) / len(recent_rewards)
        moving_average_rewards.append(moving_avg_reward)

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

# 保存模型参数
torch.save(agent.policy_net.state_dict(), os.path.join(results_dir, 'policy_net.pth'))

# 保存指标数据
np.save(os.path.join(results_dir, 'all_rewards.npy'), np.array(all_rewards))
np.save(os.path.join(results_dir, 'all_losses.npy'), np.array(all_losses))
np.save(os.path.join(results_dir, 'episode_lengths.npy'), np.array(episode_lengths))
np.save(os.path.join(results_dir, 'epsilons.npy'), np.array(epsilons))
np.save(os.path.join(results_dir, 'success_rates.npy'), np.array(success_rates))
np.save(os.path.join(results_dir, 'average_rewards.npy'), np.array(average_rewards))
np.save(os.path.join(results_dir, 'moving_average_rewards.npy'), np.array(moving_average_rewards))

print("Training completed.")
