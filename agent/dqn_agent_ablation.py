# agent/intelligent_dqn_agent.py
# 智能决策Agent：负责所有的约束检查、安全机制、动作掩码等智能决策

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class IntelligentDQNAgent:
    """
    智能DQN Agent - 负责所有智能决策逻辑
    
    职责：
    1. 基于当前状态生成有效动作掩码
    2. 实施安全机制（20%强制返航）
    3. 执行约束检查（时间、访问状态等）
    4. 选择最优动作
    5. 学习和优化策略
    
    消融实验控制：
    - use_action_mask: 是否使用动作掩码
    - use_safety_mechanism: 是否使用安全机制
    - use_constraint_check: 是否使用约束检查
    """
    
    def __init__(self, env, num_time_bins=5, hidden_size=128, learning_rate=1e-3, gamma=0.99, 
                 use_action_mask=True, use_safety_mechanism=True, use_constraint_check=True):
        self.env = env
        self.num_time_bins = num_time_bins
        self.gamma = gamma
        
        # GPU device setup with multi-GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"DQN Agent using device: {self.device}")
        print(f"Available GPUs: {self.num_gpus}")
        if self.num_gpus > 1:
            print(f"Multi-GPU setup detected: {self.num_gpus} GPUs available")
        
        # 消融实验控制参数
        self.use_action_mask = use_action_mask
        self.use_safety_mechanism = use_safety_mechanism
        self.use_constraint_check = use_constraint_check
        
        # Discretize action space
        self.action_list = self._discretize_actions()
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}
        self.action_size = len(self.action_list)
        
        # Define state size
        self.state_size = self._get_state_size()
        
        # Initialize networks and move to GPU
        self.policy_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size, hidden_size).to(self.device)
        
        # Enable multi-GPU if available (DataParallel for multiple GPUs)
        if self.num_gpus > 1:
            print(f"Enabling DataParallel across {self.num_gpus} GPUs")
            self.policy_net = nn.DataParallel(self.policy_net)
            self.target_net = nn.DataParallel(self.target_net)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # 统计信息
        self.action_mask_rejections = 0
        self.safety_interventions = 0
        self.constraint_violations = 0
        
    def _discretize_actions(self):
        """离散化动作空间"""
        action_list = []
        for loc in range(self.env.m + 1):  # Including Home
            time_lower, time_upper = self.env.get_data_time_bounds(loc)
            
            if time_upper == time_lower:
                time_values = [time_lower]
            else:
                time_values = np.linspace(time_lower, time_upper, num=self.num_time_bins)
            
            for t in time_values:
                action_list.append((loc, round(float(t), 2)))
        return action_list
        
    def _get_state_size(self):
        """获取状态向量大小，天气向量大小变为3"""
        # current_location (1) + remaining_time (1) + visited (m+1) + weather (3)
        return 1 + 1 + (self.env.m + 1) + 3
        
    def state_to_tensor(self, state):
        """将状态转换为张量，天气使用one-hot编码（大小为3）"""
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        visited = state['visited']
        weather = state['weather']
        
        # Normalize remaining_time
        remaining_time /= self.env.T_max
        
        # 天气 one-hot 编码，大小为3
        weather_one_hot = np.zeros(3)
        weather_one_hot[weather] = 1
        
        # Construct state vector
        state_vector = np.concatenate((
            np.array([current_location / (self.env.m + 1)]),
            np.array([remaining_time]),
            visited,
            weather_one_hot
        ))
        return torch.FloatTensor(state_vector).to(self.device)
    
    def _is_action_physically_feasible(self, state, action):
        """检查动作的物理可行性（基础约束）"""
        loc, t_data = action
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        weather = state['weather']
        
        # 检查是否已访问
        if loc != 0 and self.env.is_location_visited(loc):
            return False, "Already visited"
        
        # 检查数据收集时间边界
        time_lower, time_upper = self.env.get_data_time_bounds(loc)
        if not (time_lower <= t_data <= time_upper):
            return False, "Data collection time out of bounds"
        
        return True, "Feasible"
    
    def _is_action_time_feasible(self, state, action):
        """检查动作的时间可行性（约束检查）"""
        if not self.use_constraint_check:
            return True, "Constraint check disabled"
            
        loc, t_data = action
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        weather = state['weather']
        
        # Agent不知道下一步的确切天气，所以它必须依赖环境提供的期望时间
        T_flight_to_next = self.env.get_expected_flight_time(current_location, loc) # 从当前位置到目标位置
        T_return = self.env.get_expected_flight_time(loc, 0) # 从目标位置返回家
        
        # 检查剩余时间约束
        total_time_needed = T_flight_to_next + t_data + T_return
        if remaining_time < total_time_needed:
            return False, "Insufficient time"
        
        return True, "Time feasible"
    
    def _should_force_return_home(self, state):
        """安全机制：检查是否应该强制返航"""
        if not self.use_safety_mechanism:
            return False, "Safety mechanism disabled"
            
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        
        # 20%安全余量规则
        if remaining_time <= 0.2 * self.env.T_max and current_location != 0:
            return True, "Safety mechanism triggered"
        
        return False, "No safety intervention needed"
    
    def _get_valid_actions(self, state):
        """获取所有有效动作的索引 (最终修正版)"""
        valid_actions = []
        rejection_reasons = {}
        
        # 1. 安全机制检查 (最高优先级)
        should_return, safety_reason = self._should_force_return_home(state)
        if should_return:
            self.safety_interventions += 1
            for idx, action in self.action_index_mapping.items():
                if action[0] == 0:
                    valid_actions.append(idx)
            return valid_actions, {"safety_intervention": safety_reason}
        
        # 2. 遍历所有动作
        for idx, action in self.action_index_mapping.items():
            # --- FIX: 将基础物理检查与 Action Mask 开关绑定 ---
            if self.use_action_mask:
                is_physical, physical_reason = self._is_action_physically_feasible(state, action)
                if not is_physical:
                    rejection_reasons[idx] = physical_reason
                    continue
            # ----------------------------------------------------
            
            # 3. 时间约束检查 (独立于Action Mask)
            is_time_ok, time_reason = self._is_action_time_feasible(state, action)
            if not is_time_ok:
                rejection_reasons[idx] = time_reason
                if self.use_constraint_check:
                    self.constraint_violations += 1
                continue
            
            valid_actions.append(idx)
        
        return valid_actions, rejection_reasons
    
    def select_action(self, state, epsilon):
        """选择动作 (修改后)"""
        
        # 检查是否需要任何形式的智能过滤
        is_intelligent_mode = self.use_action_mask or self.use_safety_mechanism or self.use_constraint_check

        if is_intelligent_mode:
            # 只要开启了任何一个智能选项，就获取有效动作列表
            # _get_valid_actions 函数内部会根据各自的开关来执行相应的检查
            valid_actions, rejection_info = self._get_valid_actions(state)
            
            if not valid_actions: # 如果列表为空
                return None  # 无有效动作

            # 记录被掩码拒绝的动作数量 (只有在 use_action_mask 为 True 时这个统计才有意义，但代码可以通用)
            self.action_mask_rejections += (self.action_size - len(valid_actions))
            
            # Epsilon-greedy 策略应用于有效动作集
            if random.random() < epsilon:
                return random.choice(valid_actions)
            else:
                with torch.no_grad():
                    state_tensor = self.state_to_tensor(state)
                    q_values = self.policy_net(state_tensor).cpu().numpy() # Move to CPU for numpy operations
                    
                    # 创建一个只包含有效动作Q值的副本
                    valid_q_values = q_values[valid_actions]
                    
                    # 从有效动作中找到Q值最大的那个
                    best_action_index_in_valid_list = np.argmax(valid_q_values)
                    
                    # 返回它在原始动作列表中的真实索引
                    return valid_actions[best_action_index_in_valid_list]
        else:
            # 标准DQN (Vanilla DQN)，对应 Model 1
            if random.random() < epsilon:
                return random.randrange(self.action_size)
            else:
                with torch.no_grad():
                    state_tensor = self.state_to_tensor(state)
                    q_values = self.policy_net(state_tensor)
                    return torch.argmax(q_values).item()
                        
    def optimize_model(self, batch):
        """优化模型"""
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        
        batch_state_tensor = torch.stack([self.state_to_tensor(s) for s in batch_state])
        batch_action_tensor = torch.LongTensor(batch_action).to(self.device)
        batch_reward_tensor = torch.FloatTensor(batch_reward).to(self.device)
        batch_next_state_tensor = torch.stack([self.state_to_tensor(s) for s in batch_next_state])
        batch_done_tensor = torch.FloatTensor(batch_done).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(batch_state_tensor)
        state_action_values = q_values.gather(1, batch_action_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_state_tensor)
        # --- FIX: 使用与 select_action 一致的逻辑 ---
        is_intelligent_mode = self.use_action_mask or self.use_safety_mechanism or self.use_constraint_check
        
        if is_intelligent_mode:
            # 对下一状态应用与当前模型配置相符的动作过滤
            next_state_values = torch.zeros(len(batch_next_state), device=self.device)
            for idx, next_state in enumerate(batch_next_state):
                # 获取在下一个状态时，真正有效的动作
                valid_actions, _ = self._get_valid_actions(next_state)
                if len(valid_actions) > 0:
                    # 只在有效动作中寻找最大Q值
                    valid_mask = torch.zeros(self.action_size, dtype=torch.bool, device=self.device)
                    valid_mask[valid_actions] = True
                    # 将无效动作的Q值设为负无穷
                    next_q_values[idx][~valid_mask] = -float('inf')
                    # 获取修正后的最大Q值
                    next_state_values[idx] = next_q_values[idx].max()
                else:
                    # 如果没有有效动作，下一个状态的价值为0
                    next_state_values[idx] = 0.0
        else:
            # 标准DQN：直接使用所有动作中的最大Q值
            next_state_values = next_q_values.max(1)[0]
        # -------------------------------------------
               
        expected_state_action_values = batch_reward_tensor + (self.gamma * next_state_values * (1 - batch_done_tensor))
        
        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'action_mask_rejections': self.action_mask_rejections,
            'safety_interventions': self.safety_interventions,
            'constraint_violations': self.constraint_violations
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.action_mask_rejections = 0
        self.safety_interventions = 0
        self.constraint_violations = 0

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x