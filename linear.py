import json
import pulp
import numpy as np

# read config
config_path = 'config/config_5.json'
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# extract config parameters
num_locations = config["num_locations"]
T_max = config["T_max"] * 0.8
weather_prob = config["weather_prob"]
P_penalty = config["P_penalty"]
T_flight_good = np.array(config["T_flight_good"])
T_flight_bad = np.array(config["T_flight_bad"])
T_data_lower = config["T_data_lower"]
T_data_upper = config["T_data_upper"]
criticality = config["criticality"]

# calculate expected flight time
T_flight_expected = (weather_prob) * T_flight_good + (1 - weather_prob) * T_flight_bad
T_flight_expected = T_flight_expected.tolist()

# generate R_data, set reward according to criticality
# assume "HC" corresponds to 10, "LC" corresponds to 2, and the reward for location 0 is 0
R_data = [0] + [10 if c == "HC" else 2 for c in criticality[1:]]

# create model
model = pulp.LpProblem("DroneRoutePlanning", pulp.LpMaximize)   

# decision variables
locations = list(range(num_locations + 1))  # 包括位置0
x = pulp.LpVariable.dicts("x", [(i, j) for i in locations for j in locations if i != j], cat='Binary')
t = pulp.LpVariable.dicts("t", [j for j in locations if j != 0], lowBound=0, cat='Continuous')
u = pulp.LpVariable.dicts("u", [j for j in locations if j != 0], lowBound=1, upBound=num_locations, cat='Integer')

# objective function
model += (
    pulp.lpSum([R_data[j] * t[j] for j in locations if j != 0]) -
    pulp.lpSum([T_flight_expected[i][j] * x[(i,j)] for i in locations for j in locations if i != j]) -
    pulp.lpSum([t[j] for j in locations if j != 0])
), "TotalRewardMinusCost"

# constraints

# visit constraint: each location is visited once
for j in locations:
    if j != 0:
        model += pulp.lpSum([x[(i, j)] for i in locations if i != j]) == 1, f"VisitOnce_{j}"

# each location is visited once
for i in locations:
    model += pulp.lpSum([x[(i, j)] for j in locations if j != i]) == 1, f"DepartOnce_{i}"

# MTZ constraint (eliminate subcycles)
for i in locations:
    if i == 0:
        continue
    for j in locations:
        if j == 0 or j == i:
            continue
        model += u[i] - u[j] + 1 <= num_locations * (1 - x[(i,j)]), f"MTZ_{i}_{j}"

# time constraint: total flight time + data collection time <= T_max
model += (
    pulp.lpSum([T_flight_expected[i][j] * x[(i,j)] for i in locations for j in locations if i != j]) +
    pulp.lpSum([t[j] for j in locations if j != 0]) <= T_max
), "TotalTime"

# 返回Home的约束
model += pulp.lpSum([x[(i,0)] for i in locations if i != 0]) == 1, "ReturnHome"
    
# 数据收集时间上下界
for j in locations:
    if j == 0:
        continue
    model += t[j] >= T_data_lower[j], f"T{j}_Lower"
    model += t[j] <= T_data_upper[j], f"T{j}_Upper"

# 自环约束（确保 x[i,i] = 0）
for i in locations:
    for j in locations:
        if i == j:
            if (i, j) in x:
                model += x[(i,j)] == 0, f"NoSelfLoop_{i}_{j}"

# 求解模型
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

# # 输出结果
# print("Status:", pulp.LpStatus[model.status])

# print("\n决策变量 x 的值 (路径):")
# for i in locations:
#     for j in locations:
#         if i != j and pulp.value(x[(i,j)]) > 0:
#             print(f"x_{i}_{j} = {pulp.value(x[(i,j)])}")

# print("\n数据收集时间 t 的值:")
# for j in locations:
#     if j != 0:
#         print(f"t_{j} = {pulp.value(t[j])}")

# print("\n顺序变量 u 的值:")
# for j in locations:
#     if j != 0:
#         print(f"u_{j} = {pulp.value(u[j])}")

print("Objective:", pulp.value(model.objective))
