import numpy as np

class RandomAgent:
    """
    An agent that selects a random action from the environment.
    """

    def __init__(self, env):
        """
        Initialize the random agent.
        """
        self.env = env

    def select_action(self, observation):
        """
        Select a valid random action based on the current observation.
        """
        remaining_time = observation['remaining_time'][0]
        visited = observation['visited']
        
        # 获取未访问的地点索引（排除Home，即索引为0的地点）
        unvisited = np.where(visited[1:] == 0)[0] + 1  # 加1因为我们排除了Home

        # 如果没有未访问的地点，或者剩余时间不足，返回Home
        if len(unvisited) == 0 or remaining_time <= 0:
            next_location = 0  # Return Home
            T_data_next = 0.0
        else:
            # 随机选择一个未访问的地点
            next_location = np.random.choice(unvisited)
            
            # 选择在数据采集时间范围内的随机时间
            T_data_min = self.env.T_data_lower[next_location]
            T_data_max = self.env.T_data_upper[next_location]
            
            # 确保T_data_min <= T_data_max
            if T_data_min > T_data_max:
                T_data_min, T_data_max = T_data_max, T_data_min

            T_data_next = np.random.uniform(T_data_min, T_data_max)
            
            # 检查剩余时间是否足够
            # 计算飞行时间到下一个地点
            weather = observation['weather']
            T_flight_to_next = self.env._get_flight_time(observation['current_location'], next_location, weather)
            # 预计返回Home的时间
            T_return = self.env._expected_return_time(next_location)
            total_time_needed = T_flight_to_next + T_data_next + T_return
            if remaining_time < total_time_needed:
                # 剩余时间不足，返回Home
                next_location = 0
                T_data_next = 0.0

        return (next_location, T_data_next)
