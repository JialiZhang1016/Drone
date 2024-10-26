import pulp
import numpy as np

# 配置参数
num_locations = 5
T_max = 320
weather_prob = 0.4
T_flight_expected = [
    [0, 32.4, 15.2, 16.6, 22.2, 20.6],
    [32.4, 0, 21.8, 17.6, 19.4, 31.8],
    [15.2, 21.8, 0, 16.0, 28.4, 24.2],
    [16.6, 17.6, 16.0, 0, 13.0, 16.8],
    [22.2, 19.4, 28.4, 13.0, 0, 19.2],
    [20.6, 31.8, 24.2, 16.8, 19.2, 0]
]
T_data_lower = [0, 18, 10, 15, 23, 20]
T_data_upper = [0, 25, 16, 21, 30, 25]
criticality = ["HC", "HC", "LC", "HC", "LC", "LC"]
R_data = [0, 10, 2, 10, 2, 2] 

# 创建模型
model = pulp.LpProblem("DroneRoutePlanning", pulp.LpMaximize)

# 决策变量
x = pulp.LpVariable.dicts("x", [(i, j) for i in range(6) for j in range(6) if i != j], cat='Binary')
t = pulp.LpVariable.dicts("t", [j for j in range(1, 6)], lowBound=0, cat='Continuous')
u = pulp.LpVariable.dicts("u", [j for j in range(1, 6)], lowBound=1, upBound=5, cat='Integer')

# 目标函数
model += (
    pulp.lpSum([R_data[j] * t[j] for j in range(1, 6)]) -
    pulp.lpSum([T_flight_expected[i][j] * x[(i,j)] for i in range(6) for j in range(6) if i != j]) -
    pulp.lpSum([t[j] for j in range(1, 6)])
), "TotalRewardMinusCost"

# 约束条件

# 访问约束：每个位置被访问一次
for j in range(1, 6):
    model += pulp.lpSum([x[(i,j)] for i in range(6) if i != j]) == 1, f"VisitOnce_{j}"

# 每个位置出发一次
for i in range(6):
    model += pulp.lpSum([x[(i,j)] for j in range(6) if j != i]) == 1, f"DepartOnce_{i}"

# MTZ约束（消除子环路）
for i in range(1, 6):
    for j in range(1, 6):
        if i != j:
            model += u[i] - u[j] + 1 <= 5 * (1 - x[(i,j)]), f"MTZ_{i}_{j}"

# 时间约束：总飞行时间 + 数据收集时间 <= T_max
model += (
    pulp.lpSum([T_flight_expected[i][j] * x[(i,j)] for i in range(6) for j in range(6) if i != j]) +
    pulp.lpSum([t[j] for j in range(1, 6)]) <= T_max
), "TotalTime"

# 返回Home的约束
model += pulp.lpSum([x[(i,0)] for i in range(1, 6)]) == 1, "ReturnHome"

# 数据收集时间上下界
model += t[1] >= 18, "T1_Lower"
model += t[1] <= 25, "T1_Upper"
model += t[2] >= 10, "T2_Lower"
model += t[2] <= 16, "T2_Upper"
model += t[3] >= 15, "T3_Lower"
model += t[3] <= 21, "T3_Upper"
model += t[4] >= 23, "T4_Lower"
model += t[4] <= 30, "T4_Upper"
model += t[5] >= 20, "T5_Lower"
model += t[5] <= 25, "T5_Upper"

# 自环约束（确保 x[i,i] = 0）
for i in range(6):
    for j in range(6):
        if i == j:
            var_name = f"x_({i},{j})"
            if (i, j) in x:
                model += x[(i,j)] == 0, f"NoSelfLoop_{i}_{j}"

# 求解模型
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

# 输出结果
print("Status:", pulp.LpStatus[model.status])
print("Objective:", pulp.value(model.objective))
# for var in model.variables():
#     if var.varValue > 0:
#         print(var.name, "=", var.varValue)
