# Drone Route Planning with Deep Q-Network (DQN)

## Project Overview
This project implements a reinforcement learning solution for autonomous drone route planning using Deep Q-Network (DQN) with intelligent decision-making components.

## Architecture

### Environment (`pure_drone_env.py`)
- **Strict Action Validation**: Validates all actions and immediately terminates episodes for violations
- **Physical Environment**: Handles state transitions and reward calculations
- **Violation Enforcement**: Invalid actions receive -1000 penalty and force episode termination
- **Validation Rules**: 
  - No revisiting non-Home locations
  - Data collection time within bounds
  - Valid location indices only
  - No staying at current location

### Agent (`agent/intelligent_dqn_agent.py`)
- **Intelligent Decision Making**: All smart components centralized in the agent
- **Components**: Action masking, safety mechanisms, constraint checking
- **Configurable**: Supports ablation studies with component enable/disable

### Key Features
1. **Action Masking**: Filters invalid actions before selection
2. **Safety Mechanism**: 20% time threshold forced return home
3. **Constraint Checking**: Real-time time constraint validation
4. **Multi-objective Optimization**: Balance data collection vs flight costs

## Configuration
- **Environment Config**: `config/realword_8/config_8.json`
- **8 Locations + Home Base**: Real-world flight time matrices
- **Weather-dependent Flight Times**: Good/bad weather scenarios
- **Critical vs Low-Critical Locations**: HC=10x reward, LC=2x reward

## Reward System
```
R_data = 10 * T_data (HC locations) or 2 * T_data (LC locations)
Cost = -1 * (T_data + T_flight)
Total_Reward = R_data + Cost
```

## Ablation Study Models
1. **Model 1**: Vanilla DQN (no intelligent components)
2. **Model 2**: DQN + Action Masking
3. **Model 3**: DQN + Safety Mechanism  
4. **Model 4**: DQN + Constraint Checking
5. **Model 5**: Complete Intelligent Agent (all components)

## Usage
```python
python corrected_ablation_experiment.py
```

## Results
- **Strict Environment**: Invalid actions now cause immediate episode termination with -1000 penalty
- **Model Performance**: Intelligent components (action masking, safety) dramatically reduce violation rates
- **Clear Distinction**: Agent intelligence vs environment rule enforcement
- **Validation Impact**: Models without intelligent components face high violation rates in strict environment