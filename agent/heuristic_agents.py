# agent/heuristic_agents.py

import numpy as np

class HeuristicAgent:
    """启发式智能体的基类"""
    def __init__(self, env, action_list):
        self.env = env
        self.action_list = action_list
        self.action_index_mapping = {idx: action for idx, action in enumerate(self.action_list)}

    def select_action(self, state):
        """子类需要实现此方法来选择一个动作索引"""
        raise NotImplementedError

class GreedyAgent(HeuristicAgent):
    """
    贪心智能体：在每个决策点，选择效用最高的动作。
    效用 = 数据奖励 / (飞行时间 + 收集时间)
    """
    def select_action(self, state):
        best_action_idx = -1
        max_utility = -np.inf

        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]
        
        for idx, action in self.action_index_mapping.items():
            loc_next, t_data = action
            
            # 规则1: 不能选择已访问过的地点 (除非是基地，但去基地没有直接奖励)
            if loc_next != 0 and self.env.is_location_visited(loc_next):
                continue

            # 规则2: 估算成本和收益
            t_flight_expected = self.env.get_expected_flight_time(current_location, loc_next)
            time_cost = t_flight_expected + t_data
            
            # 规则3: 必须保证有足够时间执行并返回
            t_return_expected = self.env.get_expected_flight_time(loc_next, 0)
            if remaining_time < time_cost + t_return_expected:
                continue

            # 计算数据奖励
            reward_data = 0
            if self.env.criticality[loc_next] == 'HC':
                reward_data = 10 * t_data
            else:
                reward_data = 2 * t_data
            
            # 计算效用
            utility = reward_data / time_cost if time_cost > 0 else reward_data
            
            if utility > max_utility:
                max_utility = utility
                best_action_idx = idx

        # 如果找不到任何有效动作，则默认返回基地
        if best_action_idx == -1:
            for idx, action in self.action_index_mapping.items():
                if action[0] == 0:
                    return idx # 返回第一个返航动作
        
        return best_action_idx


class RuleBasedAgent(HeuristicAgent):
    """
    规则基智能体：基于一组硬编码规则进行决策。
    """
    def select_action(self, state):
        current_location = state['current_location']
        remaining_time = state['remaining_time'][0]

        # 规则1: 安全检查 (强制返航)
        t_return_expected = self.env.get_expected_flight_time(current_location, 0)
        if remaining_time < 1.5 * t_return_expected and current_location != 0:
            for idx, action in self.action_index_mapping.items():
                if action[0] == 0:
                    return idx # 强制返航
        
        # 寻找候选动作
        candidate_actions = []
        for idx, action in self.action_index_mapping.items():
            loc_next, t_data = action
            
            if loc_next == 0 or self.env.is_location_visited(loc_next):
                continue
            
            t_flight_expected = self.env.get_expected_flight_time(current_location, loc_next)
            time_cost = t_flight_expected + t_data
            t_return_expected_future = self.env.get_expected_flight_time(loc_next, 0)

            if remaining_time >= time_cost + t_return_expected_future:
                candidate_actions.append((idx, action))

        if not candidate_actions:
            for idx, action in self.action_index_mapping.items():
                if action[0] == 0:
                    return idx # 无可选动作，返航

        # 规则2: 优先访问高价值(HC)地点
        hc_actions = [a for a in candidate_actions if self.env.criticality[a[1][0]] == 'HC']
        
        if hc_actions:
            target_actions = hc_actions
        else: # 规则3: 如果没有HC地点，则访问低价值(LC)地点
            target_actions = [a for a in candidate_actions if self.env.criticality[a[1][0]] == 'LC']

        # 规则4: 在候选者中选择效用最高的
        best_action_idx = None
        max_utility = -np.inf
        for idx, action in target_actions:
            loc_next, t_data = action
            t_flight_expected = self.env.get_expected_flight_time(current_location, loc_next)
            time_cost = t_flight_expected + t_data
            
            reward_data = 10 * t_data if self.env.criticality[loc_next] == 'HC' else 2 * t_data
            utility = reward_data / time_cost if time_cost > 0 else reward_data
            
            if utility > max_utility:
                max_utility = utility
                best_action_idx = idx
        
        # 返回最佳动作索引
        if best_action_idx is not None:
            return best_action_idx
        
        # 最后防线：返航
        for idx, action in self.action_index_mapping.items():
            if action[0] == 0:
                return idx