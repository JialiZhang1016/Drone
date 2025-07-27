import json
import random
import os

def generate_config(
    num_locations=8,
    T_max=3000, # 建议增加总时间以应对更严酷的环境
    weather_prob=0.6, # 好天气概率
    P_penalty=10000,
    seed=42
):
    """
    生成配置文件，采用“基准+随机延迟”模型
    """
    size = num_locations + 1  # include Home
    random.seed(seed)

    # 1. 生成 T_flight_good 作为唯一的基准时间矩阵
    T_flight_good = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            # 使用浮点数以增加精度
            value = round(random.uniform(20.0, 70.0), 2)
            T_flight_good[i][j] = value
            T_flight_good[j][i] = value

    # 2. 定义新的不确定性参数
    # 坏天气延迟因子范围 [min, max]
    bad_weather_delay_factor = [1.1, 1.5] 
    # 极端天气概率
    extreme_weather_prob = 0.05 
    # 极端天气延迟因子范围 [min, max]
    extreme_weather_delay_factor = [1.8, 2.5]

    # 3. 生成 T_data_lower 和 T_data_upper
    T_data_lower = [0] + [random.randint(10, 30) for _ in range(size - 1)]
    T_data_upper = [0] + [lower + random.randint(5, 10) for lower in T_data_lower[1:]]

    # 4. 设置 criticality
    criticality = ["HC"] + [random.choice(["HC", "LC"]) for _ in range(size - 1)]

    # 5. 准备新的 config 字典
    config = {
        "num_locations": num_locations,
        "T_max": T_max,
        "weather_prob": weather_prob, # 好天气概率
        "extreme_weather_prob": extreme_weather_prob, # 新增：极端天气概率
        "P_penalty": P_penalty,
        "T_flight_good": T_flight_good,
        # "T_flight_bad" 已被移除
        "bad_weather_delay_factor": bad_weather_delay_factor, # 新增
        "extreme_weather_delay_factor": extreme_weather_delay_factor, # 新增
        "T_data_lower": T_data_lower,
        "T_data_upper": T_data_upper,
        "criticality": criticality
    }

    # ... (custom_json_dumps 函数保持不变) ...
    def custom_json_dumps(config_dict):
        json_lines = ['{']
        for key, value in config_dict.items():
            if isinstance(value, list):
                if all(isinstance(item, list) for item in value):  # matrix
                    json_lines.append(f'  "{key}": [')
                    for i, row in enumerate(value):
                        row_str = ', '.join(map(str, row))
                        comma = ',' if i < len(value) - 1 else ''
                        json_lines.append(f'    [{row_str}]{comma}')
                    json_lines.append('  ],')
                else:  # simple list
                    list_str = ', '.join(json.dumps(item) for item in value)
                    json_lines.append(f'  "{key}": [{list_str}],')
            else:  # simple key-value pair
                json_lines.append(f'  "{key}": {json.dumps(value)},') # 使用json.dumps确保字符串正确引用
        
        if json_lines[-1].endswith(','):
            json_lines[-1] = json_lines[-1][:-1]
        json_lines.append('}')
        return '\n'.join(json_lines)

    json_data = custom_json_dumps(config)
    
    config_dir = os.path.join("current_version","config")
    os.makedirs(config_dir, exist_ok=True)
    
    filename = f"config_{num_locations}.json"
    filepath = os.path.join(config_dir, filename)
    with open(filepath, "w") as f:
        f.write(json_data)
    print(f"Config saved to {filepath}")

# 运行此脚本以生成新的配置文件
if __name__ == "__main__":
    # 生成 m=15 的配置
    print("Generating config for m=15...")
    generate_config(num_locations=15, T_max=5000, seed=42)
    
    # 生成 m=20 的配置
    print("Generating config for m=20...")
    generate_config(num_locations=20, T_max=8000, seed=42)
