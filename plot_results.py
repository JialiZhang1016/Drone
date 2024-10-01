import numpy as np
import matplotlib.pyplot as plt

# 加载总奖励数据
rewards = np.load('rewards.npy')

# 绘制总奖励曲线
plt.figure(figsize=(12, 6))
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance')
plt.grid(True)
plt.savefig('training_performance.png')  # 保存图像
plt.show()
