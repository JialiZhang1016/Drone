import json
import pulp
import numpy as np
import csv
import os

def solve_optimization(config_path, weather_prob):
    # Read config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Update weather probability
    config["weather_prob"] = weather_prob

    # Extract config parameters
    num_locations = config["num_locations"]
    T_max = config["T_max"] * 0.8
    weather_prob = config["weather_prob"]
    P_penalty = config["P_penalty"]
    T_flight_good = np.array(config["T_flight_good"])
    T_flight_bad = np.array(config["T_flight_bad"])
    T_data_lower = config["T_data_lower"]
    T_data_upper = config["T_data_upper"]
    criticality = config["criticality"]

    # Calculate expected flight time
    T_flight_expected = (weather_prob) * T_flight_good + (1 - weather_prob) * T_flight_bad
    T_flight_expected = T_flight_expected.tolist()

    # Generate R_data, set reward according to criticality
    # Assume "HC" corresponds to 10, "LC" corresponds to 2, and the reward for location 0 is 0
    R_data = [0] + [10 if c == "HC" else 2 for c in criticality[1:]]

    # Create model
    model = pulp.LpProblem("DroneRoutePlanning", pulp.LpMaximize)   

    # Decision variables
    locations_list = list(range(num_locations + 1))  # Including location 0
    x = pulp.LpVariable.dicts("x", [(i, j) for i in locations_list for j in locations_list if i != j], cat='Binary')
    t = pulp.LpVariable.dicts("t", [j for j in locations_list if j != 0], lowBound=0, cat='Continuous')
    u = pulp.LpVariable.dicts("u", [j for j in locations_list if j != 0], lowBound=1, upBound=num_locations, cat='Integer')

    # Objective function
    model += (
        pulp.lpSum([R_data[j] * t[j] for j in locations_list if j != 0]) -
        pulp.lpSum([T_flight_expected[i][j] * x[(i,j)] for i in locations_list for j in locations_list if i != j]) -
        pulp.lpSum([t[j] for j in locations_list if j != 0])
    ), "TotalRewardMinusCost"

    # Constraints

    # Visit constraint: each location is visited once
    for j in locations_list:
        if j != 0:
            model += pulp.lpSum([x[(i, j)] for i in locations_list if i != j]) == 1, f"VisitOnce_{j}"

    # Depart constraint: each location departs once
    for i in locations_list:
        model += pulp.lpSum([x[(i, j)] for j in locations_list if j != i]) == 1, f"DepartOnce_{i}"

    # MTZ constraint (eliminate subcycles)
    for i in locations_list:
        if i == 0:
            continue
        for j in locations_list:
            if j == 0 or j == i:
                continue
            model += u[i] - u[j] + 1 <= num_locations * (1 - x[(i,j)]), f"MTZ_{i}_{j}"

    # Time constraint: total flight time + data collection time <= T_max
    model += (
        pulp.lpSum([T_flight_expected[i][j] * x[(i,j)] for i in locations_list for j in locations_list if i != j]) +
        pulp.lpSum([t[j] for j in locations_list if j != 0]) <= T_max
    ), "TotalTime"

    # Return to home constraint
    model += pulp.lpSum([x[(i,0)] for i in locations_list if i != 0]) == 1, "ReturnHome"
        
    # Data collection time bounds
    for j in locations_list:
        if j == 0:
            continue
        model += t[j] >= T_data_lower[j], f"T{j}_Lower"
        model += t[j] <= T_data_upper[j], f"T{j}_Upper"

    # No self-loop constraint
    for i in locations_list:
        for j in locations_list:
            if i == j and (i, j) in x:
                model += x[(i,j)] == 0, f"NoSelfLoop_{i}_{j}"

    # Solve the model
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    # Get the objective value
    objective_value = pulp.value(model.objective)

    return objective_value

if __name__ == "__main__":
    import sys

    # Locations and weather probabilities to iterate over
    locations = ["config/config_5.json", "config/config_10.json", "config/config_20.json"]
    weather_probs = [0.2, 0.5, 0.8, 1.0]

    # Initialize a list to store summary data
    summary_data = []

    for config_path in locations:
        # Extract location number from the config filename
        location_num = int(config_path.split('_')[-1].split('.')[0])
        for wp in weather_probs:
            # Solve the optimization problem
            objective_value = solve_optimization(config_path, wp)

            # Append the results to summary_data
            summary_data.append({
                'locations': location_num,
                'p': wp,
                'Objective': objective_value
            })

            # Optionally, print the summary for each setting
            print(f"Location: {location_num}, Weather Prob: {wp}, Objective: {objective_value:.2f}")

    # Save the summary_data to a CSV file
    results_dir = 'runs/compare_p'
    os.makedirs(results_dir, exist_ok=True)
    summary_csv_path = os.path.join(results_dir, 'summary_results_LP.csv')
    with open(summary_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['locations', 'p', 'Objective']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for data in summary_data:
            writer.writerow(data)
    
    print(f"\nSummary of results saved to {summary_csv_path}")
