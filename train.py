import gymnasium as gym
from drone_env import DroneRoutePlanningEnv
from dqn_agent import DQNAgent
import json
import torch
import random
import numpy as np
from collections import deque

# 加载配置文件
with open('config.json', 'r') as config_file:
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
epsilon_end = 0.01
epsilon_decay = 300
target_update_freq = 10
memory_size = 10000
gamma = 0.99

memory = deque(maxlen=memory_size)
steps_done = 0
all_rewards = []  # 用于记录每个 Episode 的总奖励

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
    
    for t in range(100):  # 限制每个 Episode 的最大步数
        steps_done += 1
        action_idx = agent.select_action(state, epsilon)
        if action_idx is None:
            break  # 无有效动作，结束 Episode
        action = agent.action_index_mapping[action_idx]
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # 存储经验
        memory.append((state, action_idx, reward, next_state, done))
        
        state = next_state
        
        # 经验回放
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            agent.optimize_model(batch)
        
        if done:
            break
    
    # 更新目标网络
    if episode % target_update_freq == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    all_rewards.append(total_reward)  # 记录总奖励
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# 保存模型参数
torch.save(agent.policy_net.state_dict(), 'policy_net.pth')

# 保存总奖励列表
np.save('rewards.npy', np.array(all_rewards))

print("Training completed.")
