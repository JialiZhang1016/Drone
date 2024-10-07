import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# python plot_results.py runs/5_500_20241007-004833 

# 获取结果目录路径
if len(sys.argv) > 1:
    results_dir = sys.argv[1]
else:
    print("请提供结果目录路径，例如：python plot_analysis.py runs/3_500_20231007-123456")
    sys.exit(1)

# 加载数据
all_rewards = np.load(os.path.join(results_dir, 'all_rewards.npy'))
all_losses = np.load(os.path.join(results_dir, 'all_losses.npy'))
episode_lengths = np.load(os.path.join(results_dir, 'episode_lengths.npy'))
epsilons = np.load(os.path.join(results_dir, 'epsilons.npy'))
success_rates = np.load(os.path.join(results_dir, 'success_rates.npy'))
average_rewards = np.load(os.path.join(results_dir, 'average_rewards.npy'))
moving_average_rewards = np.load(os.path.join(results_dir, 'moving_average_rewards.npy'))

# 绘制并保存图像
def plot_and_save(data, ylabel, filename, x=None):
    plt.figure()
    if x is None:
        plt.plot(data)
    else:
        plt.plot(x, data)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.savefig(os.path.join(results_dir, f"{filename}.png"))
    plt.close()

# 绘制所有奖励
plot_and_save(all_rewards, 'Total Reward per Episode', 'all_rewards')

# 绘制所有损失
plot_and_save(all_losses, 'Average Loss per Episode', 'all_losses')

# 绘制Episode长度
plot_and_save(episode_lengths, 'Episode Length', 'episode_lengths')

# 绘制Epsilon值
plot_and_save(epsilons, 'Epsilon', 'epsilons')

# 绘制成功率
success_rate_interval = 10  # 与训练代码中保持一致
success_rate_episodes = np.arange(success_rate_interval, len(success_rates)*success_rate_interval+1, success_rate_interval)
plot_and_save(success_rates, 'Success Rate', 'success_rates', x=success_rate_episodes)

# 绘制平均奖励
average_reward_interval = 20  # 与训练代码中保持一致
average_reward_episodes = np.arange(average_reward_interval, len(average_rewards)*average_reward_interval+1, average_reward_interval)
plot_and_save(average_rewards, 'Average Reward', 'average_rewards', x=average_reward_episodes)

# 绘制移动平均奖励
moving_average_interval = 15  # 与训练代码中保持一致
moving_avg_reward_episodes = np.arange(moving_average_interval-1, len(moving_average_rewards)+moving_average_interval-1)
plot_and_save(moving_average_rewards, 'Moving Average Reward', 'moving_average_rewards', x=moving_avg_reward_episodes)

print("已在目录中保存所有图表：", results_dir)
