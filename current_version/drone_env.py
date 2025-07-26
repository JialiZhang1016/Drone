# pure_drone_env.py
# 纯粹的物理环境：只负责状态转换和奖励计算，不做任何智能决策

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PureDroneRoutePlanningEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, use_reward_shaping=True):
        super(PureDroneRoutePlanningEnv, self).__init__()
        
        # --- 配置加载 ---
        self.m = config['num_locations']
        self.T_max = config['T_max']
        self.P_penalty = config['P_penalty']
        
        # Reward shaping 控制标志
        self.use_reward_shaping = use_reward_shaping
        
        # 天气概率
        self.p_good = config['weather_prob']
        self.p_extreme = config['extreme_weather_prob']
        self.p_bad = 1.0 - self.p_good - self.p_extreme
        
        # 飞行时间模型
        self.T_flight_good = np.array(config['T_flight_good'])
        self.bad_delay_range = config['bad_weather_delay_factor']
        self.extreme_delay_range = config['extreme_weather_delay_factor']
        
        self.T_data_lower = np.array(config['T_data_lower'])
        self.T_data_upper = np.array(config['T_data_upper'])
        self.criticality = config['criticality']
        
        # --- 空间定义 ---
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.m + 1),
            spaces.Box(low=self.T_data_lower, high=self.T_data_upper, dtype=np.float32)
        ))
        
        # 观测空间：天气状态现在是3种 (0: Good, 1: Bad, 2: Extreme)
        self.observation_space = spaces.Dict({
            'current_location': spaces.Discrete(self.m + 1),
            'remaining_time': spaces.Box(low=-np.inf, high=self.T_max, shape=(1,), dtype=np.float32),
            'visited': spaces.MultiBinary(self.m + 1),
            'weather': spaces.Discrete(3) 
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.L_t = 0
        self.T_t_rem = self.T_max
        self.V_t = np.zeros(self.m + 1, dtype=np.int8)
        self.V_t[0] = 1
        
        # 根据新的三状态天气模型生成天气
        self.W_t = np.random.choice([0, 1, 2], p=[self.p_good, self.p_bad, self.p_extreme])
        
        self.visit_order = [0]
        self.done = False
        self.total_reward = 0.0
        self.timestep = 0
        self.safety_violations = 0
        self.constraint_violations = 0
        return self._get_observation(), {}
    
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has finished. Call reset() to start a new episode.")
        
        L_next, T_data_next = action
        L_next = int(L_next)
        T_data_next = float(T_data_next)
        
        # 1. 更新天气状态
        self.W_t = np.random.choice([0, 1, 2], p=[self.p_good, self.p_bad, self.p_extreme])
        
        # 2. 动态计算飞行时间
        T_flight_to_next = self._get_flight_time(self.L_t, L_next, self.W_t)
        
        # 3. 计算奖励 (基于计划的 T_data_next)
        reward = self._calculate_reward(L_next, T_data_next, T_flight_to_next)
        self.total_reward += reward
        
        # 4. 更新状态
        self.T_t_rem -= (T_flight_to_next + T_data_next)
        if self.T_t_rem < 0:
            self.constraint_violations += 1
        
        self.L_t = L_next
        self.V_t[L_next] = 1
        self.visit_order.append(L_next)
        
        observation = self._get_observation()
        
        # 5. 处理回合结束和最终惩罚
        self.done = self._check_done()
        if self.done and self.L_t != 0:
            reward -= self.P_penalty
            self.total_reward -= self.P_penalty
            self.safety_violations += 1

        self.timestep += 1
        info = { 'reward': reward } # Info可以简化
        return observation, reward, self.done, info
    
    def _get_flight_time(self, L_from, L_to, weather):
        """动态、随机地计算飞行时间"""
        base_time = self.T_flight_good[L_from, L_to]
        if weather == 0:  # 好天气
            return base_time
        elif weather == 1:  # 坏天气
            delay_factor = np.random.uniform(self.bad_delay_range[0], self.bad_delay_range[1])
            return base_time * delay_factor
        else:  # 极端天气 (weather == 2)
            delay_factor = np.random.uniform(self.extreme_delay_range[0], self.extreme_delay_range[1])
            return base_time * delay_factor

    def _expected_return_time(self, L_from):
        """计算包含所有天气情况的期望返回时间"""
        T_return_good = self.T_flight_good[L_from, 0]
        
        # 计算坏天气和极端天气下的平均飞行时间
        avg_bad_delay = (self.bad_delay_range[0] + self.bad_delay_range[1]) / 2
        T_return_bad_avg = T_return_good * avg_bad_delay
        
        avg_extreme_delay = (self.extreme_delay_range[0] + self.extreme_delay_range[1]) / 2
        T_return_extreme_avg = T_return_good * avg_extreme_delay
        
        # 根据概率加权求和
        T_expected_return = (self.p_good * T_return_good) + \
                            (self.p_bad * T_return_bad_avg) + \
                            (self.p_extreme * T_return_extreme_avg)
        return T_expected_return

    def _get_observation(self):
        return {
            'current_location': self.L_t,
            'remaining_time': np.array([self.T_t_rem], dtype=np.float32),
            'visited': self.V_t.copy(),
            'weather': self.W_t
        }

    def _calculate_reward(self, L_next, T_data_next, T_flight_to_next):
        R_data = 0
        if self.V_t[L_next] == 0:
            if self.criticality[L_next] == 'HC':
                R_data = 10 * T_data_next
            else:
                R_data = 2 * T_data_next
        
        C = -1 * (T_data_next + T_flight_to_next)
        base_reward = R_data + C
        
        if self.use_reward_shaping:
            # 将惩罚改为奖励
            safety_bonus = self._calculate_safety_bonus()
            return base_reward + safety_bonus  # 注意这里是加号
        else:
            return base_reward

    def _calculate_danger_penalty(self):
        T_return_home = self._expected_return_time(self.L_t)
        safety_margin = 1.5
        danger_ratio = (T_return_home * safety_margin) / self.T_t_rem if self.T_t_rem > 0 else float('inf')
        if danger_ratio > 0.8:
            penalty_coefficient = 50
            penalty = penalty_coefficient * (danger_ratio - 0.8)**2
        else:
            penalty = 0
        return penalty

    def _calculate_safety_bonus(self):
        """计算一个基于安全状态的奖励"""
        # 如果不在家，计算安全奖励
        if self.L_t != 0 and self.T_t_rem > 0:
            T_return_home = self._expected_return_time(self.L_t)
            
            # 剩余时间与返航时间的安全比率
            # 如果剩余时间是返航时间的3倍，则 safety_ratio 约为 0.66
            safety_ratio = T_return_home / self.T_t_rem
            
            # 当 safety_ratio 很小时（非常安全），给予更高奖励
            # 例如，使用 1 - safety_ratio，并乘以一个系数
            bonus_coefficient = 0.5  # 调整此系数
            bonus = bonus_coefficient * max(0, 1 - safety_ratio)
            return bonus
        return 0
    
    def _check_done(self):
        if self.L_t == 0 and self.timestep > 0:
            return True
        if self.T_t_rem <= 0:
            return True
        if np.all(self.V_t == 1):
            return True
        return False
    
    # 辅助查询接口
    def get_expected_return_time(self, L_from):
        return self._expected_return_time(L_from)
        
    def is_location_visited(self, location):
        return self.V_t[location] == 1
        
    def get_data_time_bounds(self, location):
        return self.T_data_lower[location], self.T_data_upper[location]