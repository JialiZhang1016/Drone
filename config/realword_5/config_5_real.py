import json
import pandas as pd
from geopy.distance import geodesic
import random

# 读取xlsx文件并提取位置信息
df = pd.read_excel('config/RL_5_locations.xlsx')
locations = df[['Names', 'Longitude,latitude']].to_dict('records')

# 计算两点之间的地理距离
def calculate_distance(loc1, loc2):
    coord1 = tuple(map(float, loc1['Longitude,latitude'].split(',')))
    coord2 = tuple(map(float, loc2['Longitude,latitude'].split(',')))
    return geodesic(coord1, coord2).meters / 8
# time = distance in meters / 10m/s 

# 创建一个6x6距离矩阵
distance_matrix = []
for loc1 in locations:
    row = []
    for loc2 in locations:
        if loc1 == loc2:
            row.append(0)  # 对角线值为0
        else:
            row.append(round(calculate_distance(loc1, loc2), 2))  # 保留两位小数
    distance_matrix.append(row)

# 保持对称性
for i in range(len(distance_matrix)):
    for j in range(i, len(distance_matrix)):
            distance_matrix[j][i] = distance_matrix[i][j]

# 打印矩阵（可选）
print("T_flight_good (in seconds):")
for row in distance_matrix:
    print(row)

# 将距离矩阵存入config/config_5_real.json文件中
json_file = 'config/config_5_real.json'

with open(json_file, 'r') as file:
    config_data = json.load(file)

# 替换T_flight_good部分的内容
config_data['T_flight_good'] = distance_matrix

# 替换T_flight_bad部分的内容
T_flight_bad = []
for i, row in enumerate(distance_matrix):
    new_row = []
    for j, d in enumerate(row):
        if i == j:
            new_row.append(0)  # 对角线为0
        else:
            new_row.append(round(d + random.randint(10, 50), 2))  # 加上随机增量
    T_flight_bad.append(new_row)

config_data['T_flight_bad'] = T_flight_bad

print("T_flight_bad (in seconds):")
for row in T_flight_bad:
    print(row)

# 将修改后的数据写回json文件
with open(json_file, 'w') as file:
    json.dump(config_data, file, indent=4)

print("Updated config/config_5_real.json successfully.")
