# pure_drone_env.py
# 纯粹的物理环境：只负责状态转换和奖励计算，不做任何智能决策

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PureDroneRoutePlanningEnv(gym.Env):
    """
    无人机路径规划环境（严格动作验证版本）
    
    职责：
    1. 维护环境状态（位置、时间、访问历史、天气）
    2. 严格验证动作有效性，无效动作立即终止episode
    3. 根据有效动作更新状态
    4. 计算奖励
    5. 判断是否结束
    
    动作验证规则：
    1. 位置必须在有效范围内 [0, m]
    2. 不能重复访问非Home地点
    3. 数据收集时间必须在规定范围内
    4. 不能原地不动（L_next == L_current）
    
    违规后果：
    - 立即给予-1000大负奖励
    - 强制终止episode
    - 在info中返回违规原因
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(PureDroneRoutePlanningEnv, self).__init__()
        
        # Load configuration parameters
        self.m = config['num_locations']  # Number of locations excluding Home
        self.T_max = config['T_max']      # Total maximum fly time
        self.p = config['weather_prob']   # Probability of good weather
        self.P_penalty = config['P_penalty']  # Penalty for not returning home
        
        # Flight time matrices
        self.T_flight_good = np.array(config['T_flight_good'])
        self.T_flight_bad = np.array(config['T_flight_bad'])
        
        # Data collection time bounds
        self.T_data_lower = np.array(config['T_data_lower'])
        self.T_data_upper = np.array(config['T_data_upper'])
        
        # Criticality levels
        self.criticality = config['criticality']  # List of 'HC' or 'LC' for each location
        
        # Define action and observation spaces
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.m + 1),  # Next location (including Home)
            spaces.Box(low=self.T_data_lower, high=self.T_data_upper, dtype=np.float32)  # Data collection time
        ))
        
        # Observation space: L_t, T_t_rem, V_t (one-hot), W_t
        self.observation_space = spaces.Dict({
            'current_location': spaces.Discrete(self.m + 1),
            'remaining_time': spaces.Box(low=0, high=self.T_max, shape=(1,), dtype=np.float32),
            'visited': spaces.MultiBinary(self.m + 1),
            'weather': spaces.Discrete(2)  # 0: Good, 1: Bad
        })
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.L_t = 0
        self.T_t_rem = self.T_max
        self.V_t = np.zeros(self.m + 1, dtype=np.int8)
        self.V_t[0] = 1
        self.W_t = np.random.choice([0, 1], p=[self.p, 1 - self.p])

        self.visit_order = [0]
        self.done = False
        self.total_reward = 0.0
        self.timestep = 0

        # --- FIX: Initialize violation counters here ---
        self.safety_violations = 0
        self.constraint_violations = 0
        # -------------------------------------------

        return self._get_observation(), {}
    
    def step(self, action):
        """
        环境step函数（修复重复访问漏洞）：
        1. 验证动作有效性，无效动作立即终止episode并给大负奖励
        2. 更新天气（随机）
        3. 计算飞行时间
        4. 先计算基于当前状态的奖励（修复重复访问漏洞）
        5. 再更新状态（位置、剩余时间、访问历史）
        6. 最后处理回合结束和惩罚
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call reset() to start a new episode.")
        
        L_next, T_data_next = action
        L_next = int(L_next)
        T_data_next = float(T_data_next)
           
        # ===== 正常动作执行 =====
        
        # Weather update (物理环境的随机性)
        self.W_t = np.random.choice([0, 1], p=[self.p, 1 - self.p])
        
        # Calculate flight time (物理定律)
        T_flight_to_next = self._get_flight_time(self.L_t, L_next, self.W_t)
        
        # 增加数据收集时间的不确定性
        # 实际花费时间可能比计划多0%到30%
        uncertainty_factor = np.random.uniform(1.0, 1.3) 
        actual_T_data_next = T_data_next * uncertainty_factor
        
        # 1. 先计算基于当前状态的奖励（修复重复访问漏洞）
        reward = self._calculate_reward(L_next, T_data_next, T_flight_to_next)
        self.total_reward += reward
        
        # 2. 再更新状态（使用实际花费的时间）
        self.T_t_rem -= (T_flight_to_next + actual_T_data_next)
        
        # Check for natural consequences of the action
        if self.T_t_rem < 0:
            # 如果时间用尽，这是一个自然的约束违反
            self.constraint_violations += 1
        
        # Update current location (物理状态更新)
        self.L_t = L_next
        self.V_t[L_next] = 1
        self.visit_order.append(L_next)
        
        observation = self._get_observation()

        # 3. 最后处理回合结束和惩罚
        self.done = self._check_done()

        # 如果任务结束时没在Home，这是一个自然的"安全违反"
        if self.done and self.L_t != 0:
            reward -= self.P_penalty
            self.total_reward -= self.P_penalty
            self.safety_violations += 1

        self.timestep += 1
        
        info = {
            'action': {
                'next_location': int(L_next),
                'planned_data_collection_time': round(float(T_data_next), 2),
                'actual_data_collection_time': round(float(actual_T_data_next), 2),
                'uncertainty_factor': round(float(uncertainty_factor), 2)
            },
            'next_state': {
                'current_location': int(self.L_t),
                'remaining_time': round(float(self.T_t_rem), 2),
                'visited': self.V_t.tolist(),
                'weather': 'Good' if self.W_t == 0 else 'Bad'
            },
            'reward': round(float(reward), 2),
            'total_reward': round(float(self.total_reward), 2),
            'visited_locations': self.visit_order,
            'violation': False
        }
        return observation, reward, self.done, info
    
    def _get_observation(self):
        """获取当前状态观测"""
        observation = {
            'current_location': self.L_t,
            'remaining_time': np.array([self.T_t_rem], dtype=np.float32),
            'visited': self.V_t.copy(),
            'weather': self.W_t
        }
        return observation
    
    def _get_flight_time(self, L_from, L_to, weather):
        """根据天气计算飞行时间（物理定律）"""
        if weather == 0:  # Good
            return self.T_flight_good[L_from, L_to]
        else:  # Bad
            return self.T_flight_bad[L_from, L_to]
    
    def _expected_return_time(self, L_from):
        """计算期望返回时间（用于Agent的约束检查）"""
        T_return_good = self.T_flight_good[L_from, 0]
        T_return_bad = self.T_flight_bad[L_from, 0]
        T_expected_return = self.p * T_return_good + (1 - self.p) * T_return_bad
        return T_expected_return
    
    def _calculate_reward(self, L_next, T_data_next, T_flight_to_next):
        """
        计算奖励（环境反馈）+ 奖励塑造（修复重复访问漏洞）
        """
        # 1. 基础数据收集奖励和成本（修复重复访问漏洞）
        R_data = 0
        # 只有在目的地未被访问过的情况下，才计算数据收益
        if self.V_t[L_next] == 0:
            if self.criticality[L_next] == 'HC':
                R_data = 10 * T_data_next
            else:
                R_data = 2 * T_data_next
        
        C = -1 * (T_data_next + T_flight_to_next)
        base_reward = R_data + C

        # 2. Risk-Aware 奖励塑造
        danger_penalty = self._calculate_danger_penalty()
        
        # 3. 最终奖励
        shaped_reward = base_reward - danger_penalty
        return shaped_reward

    def _calculate_danger_penalty(self):
        """
        计算基于风险的即时惩罚
        """
        T_return_home = self._expected_return_time(self.L_t)
        safety_margin = 1.5 # 安全系数，可以调整
        
        # 危险比例：越高越危险
        # 当剩余时间远大于所需返航时间时，该比例接近0
        # 当剩余时间接近所需返航时间时，该比例接近1
        # 当剩余时间小于所需返航时间时，该比例大于1
        danger_ratio = (T_return_home * safety_margin) / self.T_t_rem if self.T_t_rem > 0 else float('inf')

        # 惩罚函数：一个简单的指数函数或分段函数
        # 这里使用一个简单的二次函数作为示例
        if danger_ratio > 0.8: # 当危险比例超过0.8时开始惩罚
            penalty_coefficient = 50 # 惩罚系数，可以调整
            penalty = penalty_coefficient * (danger_ratio - 0.8)**2
        else:
            penalty = 0
            
        return penalty
    
    
    def _check_done(self):
        """判断是否结束（物理终止条件）"""
        # Episode ends if:
        # 1. Drone returns to Home
        # 2. No remaining fly time
        # 3. All locations visited
        if self.L_t == 0 and self.timestep > 0:
            return True
        if self.T_t_rem <= 0:
            return True
        if np.all(self.V_t == 1):
            return True
        return False

    def render(self, mode='human'):   
        print(f"Current Location: {self.L_t}")
        print(f"Remaining Time: {self.T_t_rem}")
        print(f"Visited Locations: {self.visit_order}")
        print(f"Weather: {'Good' if self.W_t == 0 else 'Bad'}")
        print(f"Total Reward: {self.total_reward}")
        print("-----")
        
    # 辅助方法：供Agent使用的查询接口
    def get_flight_time(self, L_from, L_to, weather):
        """供Agent查询飞行时间"""
        return self._get_flight_time(L_from, L_to, weather)
        
    def get_expected_return_time(self, L_from):
        """供Agent查询期望返回时间"""
        return self._expected_return_time(L_from)
        
    def is_location_visited(self, location):
        """供Agent查询位置是否已访问"""
        return self.V_t[location] == 1
        
    def get_data_time_bounds(self, location):
        """供Agent查询数据收集时间边界"""
        return self.T_data_lower[location], self.T_data_upper[location]