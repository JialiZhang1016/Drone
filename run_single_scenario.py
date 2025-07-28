#!/usr/bin/env python3
"""
Single scenario runner for parallel execution
Usage: python run_single_scenario.py <scenario_index>
"""

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
    """Generate and save a specific experiment configuration file."""
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

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write the configuration file
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Successfully saved config to {filepath}")

def run_single_scenario(scenario_index):
    """Run a single scenario by index"""
    
    # Define all experiment cases
    EXPERIMENT_CASES = [
        # Group 1: Core Scaling Analysis (Stable Weather)
        {'name': '5_loc_stable',   'locations': 5,  'weather_prob': 0.8, 'extreme_weather_prob': 0.05},
        {'name': '8_loc_stable',   'locations': 8,  'weather_prob': 0.8, 'extreme_weather_prob': 0.05},
        {'name': '15_loc_stable',  'locations': 15, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05},
        {'name': '20_loc_stable',  'locations': 20, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05},
        
        # Group 2: Robustness to Uncertainty (Unstable Weather)
        {'name': '5_loc_unstable',  'locations': 5,  'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
        {'name': '8_loc_unstable',  'locations': 8,  'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
        {'name': '15_loc_unstable', 'locations': 15, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
        {'name': '20_loc_unstable', 'locations': 20, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
        
        # Group 3: Stress Test (Highly Unstable & Extreme Conditions)
        {'name': '15_loc_extreme', 'locations': 15, 'weather_prob': 0.4, 'extreme_weather_prob': 0.20},
        {'name': '20_loc_extreme', 'locations': 20, 'weather_prob': 0.4, 'extreme_weather_prob': 0.25},
    ]
    
    if scenario_index >= len(EXPERIMENT_CASES):
        print(f"Error: Scenario index {scenario_index} out of range (0-{len(EXPERIMENT_CASES)-1})")
        sys.exit(1)
    
    case = EXPERIMENT_CASES[scenario_index]
    num_loc = case['locations']
    weather_p = case['weather_prob']
    extreme_p = case['extreme_weather_prob']
    
    # Calculate parameters
    t_max = 2500 + (num_loc - 5) * 400
    p_penalty = int(round(8000 * (num_loc / 8.0)**1.5 / 1000) * 1000)
    
    print("=" * 80)
    print(f"PARALLEL SCENARIO {scenario_index}: {case['name']}")
    print(f"Parameters: Locations={num_loc}, WeatherProb={weather_p}, ExtremeProb={extreme_p}")
    print(f"Calculated: T_max={t_max}, P_penalty={p_penalty}")
    print("=" * 80)
    
    # Generate configuration file
    config_output_dir = "generated_configs_parallel"
    config_filename = f"config_loc{num_loc}_wp{weather_p}_ewp{extreme_p}.json"
    config_filepath = os.path.join(config_output_dir, config_filename)
    
    generate_config_for_experiment(
        filepath=config_filepath,
        num_locations=num_loc,
        T_max=t_max,
        weather_prob=weather_p,
        P_penalty=p_penalty,
        extreme_weather_prob=extreme_p,
        seed=42
    )
    
    # Run the experiment
    try:
        run_single_experiment(config_filepath)
        print(f"✅ Successfully completed scenario {scenario_index}: {case['name']}")
    except Exception as e:
        print(f"❌ Scenario {scenario_index} failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_single_scenario.py <scenario_index>")
        print("Available scenarios (0-9):")
        scenarios = [
            "5_loc_stable", "8_loc_stable", "15_loc_stable", "20_loc_stable",
            "5_loc_unstable", "8_loc_unstable", "15_loc_unstable", "20_loc_unstable", 
            "15_loc_extreme", "20_loc_extreme"
        ]
        for i, name in enumerate(scenarios):
            print(f"  {i}: {name}")
        sys.exit(1)
    
    scenario_index = int(sys.argv[1])
    run_single_scenario(scenario_index)