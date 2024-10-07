import json
import random
import os

def generate_config(
    num_locations=5,
    T_max=400,
    weather_prob=0.4,
    P_penalty=10000,
    seed=42
):
    size = num_locations + 1  # include Home

    # set random seed (optional)
    random.seed(seed)

    # generate T_flight_good symmetric matrix, non-diagonal elements are random integers from 10 to 30
    T_flight_good = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            value = random.randint(10, 30)  # random integer from [10, 30]
            T_flight_good[i][j] = value
            T_flight_good[j][i] = value

    # generate T_flight_bad symmetric matrix, non-diagonal elements are T_flight_good + [5,10]
    T_flight_bad = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            additional = random.randint(5, 10)  # random integer from [5, 10]
            value = T_flight_good[i][j] + additional
            T_flight_bad[i][j] = value
            T_flight_bad[j][i] = value

    # generate T_data_lower and T_data_upper
    T_data_lower = [0] + [random.randint(10, 30) for _ in range(size - 1)]
    T_data_upper = [0] + [lower + random.randint(5, 10) for lower in T_data_lower[1:]]

    # set criticality, Home is "HC", the others are randomly assigned "HC" or "LC"
    criticality = ["HC"] + [random.choice(["HC", "LC"]) for _ in range(size - 1)]

    # prepare config dictionary
    config = {
        "num_locations": num_locations,
        "T_max": T_max,
        "weather_prob": weather_prob,
        "P_penalty": P_penalty,
        "T_flight_good": T_flight_good,
        "T_flight_bad": T_flight_bad,
        "T_data_lower": T_data_lower,
        "T_data_upper": T_data_upper,
        "criticality": criticality
    }

    # custom JSON serialization to ensure internal lists are on a single line
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
                json_lines.append(f'  "{key}": {value},')
        
        # remove the last comma and close the curly braces
        if json_lines[-1].endswith(','):
            json_lines[-1] = json_lines[-1][:-1]
        json_lines.append('}')
        return '\n'.join(json_lines)

    # generate JSON string in custom format
    json_data = custom_json_dumps(config)

    # Ensure the config directory exists
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)

    # save JSON data to file in the config folder
    filename = f"config_{num_locations}.json"
    filepath = os.path.join(config_dir, filename)
    with open(filepath, "w") as f:
        f.write(json_data)

    print(f"Config saved to {filepath}")

# example usage:
generate_config(num_locations=15, T_max=600, weather_prob=0.4, P_penalty=10000)