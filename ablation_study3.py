import os
import sys
import json
import random

# Import the necessary functions from your existing scripts
try:
    from ablation_study2 import run_single_experiment
except ImportError:
    print("Error: Could not import 'run_single_experiment' from 'ablation_study2.py'.")
    print("Please ensure all scripts are in the same directory.")
    sys.exit(1)

def generate_config_for_experiment(filepath, num_locations, T_max, weather_prob, P_penalty, extreme_weather_prob, seed=42):
    """
    Generates and saves a specific experiment configuration file.
    
    This function is based on the logic from generate_m_data.py but is adapted
    to be more flexible for orchestration.

    Args:
        filepath (str): The full path where the JSON config file will be saved.
        All other arguments correspond to environment parameters.
    """
    print(f"Generating configuration file: {filepath}")
    size = num_locations + 1  # include Home
    random.seed(seed)

    # 1. Generate T_flight_good as the baseline flight time matrix
    T_flight_good = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            value = round(random.uniform(20.0, 70.0), 2)
            T_flight_good[i][j] = value
            T_flight_good[j][i] = value

    # 2. Define uncertainty parameters
    bad_weather_delay_factor = [1.1, 1.5]
    extreme_weather_delay_factor = [1.8, 2.5]

    # 3. Generate data collection time ranges
    T_data_lower = [0] + [random.randint(10, 30) for _ in range(size - 1)]
    T_data_upper = [0] + [lower + random.randint(5, 10) for lower in T_data_lower[1:]]

    # 4. Set criticality levels
    criticality = ["HC"] + [random.choice(["HC", "LC"]) for _ in range(size - 1)]

    # 5. Assemble the configuration dictionary
    config = {
        "num_locations": num_locations,
        "T_max": T_max,
        "weather_prob": weather_prob,
        "extreme_weather_prob": extreme_weather_prob,
        "P_penalty": P_penalty,
        "T_flight_good": T_flight_good,
        "bad_weather_delay_factor": bad_weather_delay_factor,
        "extreme_weather_delay_factor": extreme_weather_delay_factor,
        "T_data_lower": T_data_lower,
        "T_data_upper": T_data_upper,
        "criticality": criticality
    }

    # Custom pretty-printer for JSON to maintain readability
    def custom_json_dumps(config_dict):
        json_lines = ['{']
        items = list(config_dict.items())
        for i, (key, value) in enumerate(items):
            line_end = ',' if i < len(items) - 1 else ''
            if isinstance(value, list) and all(isinstance(item, list) for item in value):  # Matrix
                json_lines.append(f'  "{key}": [')
                for j, row in enumerate(value):
                    row_str = ', '.join(map(str, row))
                    row_end = ',' if j < len(value) - 1 else ''
                    json_lines.append(f'    [{row_str}]{row_end}')
                json_lines.append(f'  ]{line_end}')
            else:  # Simple key-value or list
                json_lines.append(f'  "{key}": {json.dumps(value)}{line_end}')
        json_lines.append('}')
        return '\n'.join(json_lines)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write the configuration file
    with open(filepath, "w") as f:
        f.write(custom_json_dumps(config))
    print(f"Successfully saved config to {filepath}")


def run_experiment_suite(experiment_cases):
    """
    Orchestrates a full suite of experiments based on a list of cases.
    """
    print("=" * 80)
    print("Starting Automated Experiment Suite")
    print(f"Total scenarios to run: {len(experiment_cases)}")
    print("=" * 80)

    config_output_dir = "generated_configs"

    for i, case in enumerate(experiment_cases):
        num_loc = case['locations']
        weather_p = case['weather_prob']
        extreme_p = case['extreme_weather_prob']

        # --- 1. Automatically calculate appropriate T_max and P_penalty ---
        # Heuristic for T_max: Start with a base time and add time for each location.
        t_max = 2500 + (num_loc - 5) * 400
        
        # Heuristic for P_penalty: Scale penalty exponentially with location count.
        p_penalty = int(round(8000 * (num_loc / 8.0)**1.5 / 1000) * 1000)

        print("\n" + "=" * 80)
        print(f"Executing Scenario {i+1}/{len(experiment_cases)}")
        print(f"  Parameters: Locations={num_loc}, GoodWeatherProb={weather_p}, ExtremeWeatherProb={extreme_p}")
        print(f"  Calculated: T_max={t_max}, P_penalty={p_penalty}")
        print("-" * 80)

        # --- 2. Generate the specific configuration file for this scenario ---
        config_filename = f"config_loc{num_loc}_wp{weather_p}_ewp{extreme_p}.json"
        config_filepath = os.path.join(config_output_dir, config_filename)

        generate_config_for_experiment(
            filepath=config_filepath,
            num_locations=num_loc,
            T_max=t_max,
            weather_prob=weather_p,
            P_penalty=p_penalty,
            extreme_weather_prob=extreme_p,
            seed=42  # Use a fixed seed for reproducible environments
        )

        # --- 3. Run the full training and evaluation pipeline ---
        try:
            run_single_experiment(config_filepath)
        except Exception as e:
            print(f"\n{'!'*20} SCENARIO FAILED {'!'*20}")
            print(f"An error occurred while running the experiment for {config_filepath}:")
            print(e)
            print("Proceeding to the next scenario.")
            continue

        print(f"\n--- Successfully Completed Scenario {i+1}/{len(experiment_cases)} ---")

    print("\n" + "=" * 80)
    print("Experiment suite finished.")
    print("Results for each run are saved in separate folders inside 'ablation_results/'.")
    print("Configuration files for each run are saved in 'generated_configs/'.")
    print("=" * 80)


if __name__ == "__main__":
    # ============================================================================
    # DEFINE YOUR EXPERIMENTAL SCENARIOS HERE
    # ============================================================================
    # Each dictionary in this list defines a unique experiment.
    # 'T_max' and 'P_penalty' will be calculated automatically.
    
    EXPERIMENT_CASES = [
        # ========================================================================
        # Group 1: Core Scaling Analysis (Stable Weather)
        # Purpose: To see how performance scales with size in a predictable environment.
        # Connects to: "Performance Scaling vs. Number of Locations" plot.
        # ========================================================================
        {'name': '5_loc_stable',   'locations': 5,  'weather_prob': 0.8, 'extreme_weather_prob': 0.05},
        {'name': '8_loc_stable',   'locations': 8,  'weather_prob': 0.8, 'extreme_weather_prob': 0.05},
        {'name': '15_loc_stable',  'locations': 15, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05},
        {'name': '20_loc_stable',  'locations': 20, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05},

        # ========================================================================
        # Group 2: Robustness to Uncertainty (Unstable Weather)
        # Purpose: To compare performance in stable vs. unstable weather at each size.
        # Connects to: "Robustness to Weather Conditions" plot.
        # ========================================================================
        {'name': '5_loc_unstable',  'locations': 5,  'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
        {'name': '8_loc_unstable',  'locations': 8,  'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
        {'name': '15_loc_unstable', 'locations': 15, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
        {'name': '20_loc_unstable', 'locations': 20, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15},

        # ========================================================================
        # Group 3: Stress Test (Highly Unstable & Extreme Conditions)
        # Purpose: To see how the agent holds up at the limits of its capability.
        # Connects to: The "difficult" end of the "Performance Heatmap".
        # ========================================================================
        {'name': '15_loc_extreme', 'locations': 15, 'weather_prob': 0.4, 'extreme_weather_prob': 0.20},
        {'name': '20_loc_extreme', 'locations': 20, 'weather_prob': 0.4, 'extreme_weather_prob': 0.25},
    ]

    # Execute the entire suite of experiments
    run_experiment_suite(EXPERIMENT_CASES)