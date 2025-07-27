# Drone Route Planning with Deep Q-Network (DQN)

## Project Overview
This project implements a reinforcement learning solution for autonomous drone route planning using Deep Q-Network (DQN) with intelligent decision-making components. The system features an ablation study framework to evaluate the contribution of different intelligent components under uncertainty conditions.

## Architecture

### Environment (`drone_env.py`)
- **Physical Environment**: Handles state transitions and reward calculations
- **Uncertainty Modeling**: Introduces random uncertainty in data collection time (1.0-1.3x factor)
- **Risk-Aware Reward Shaping**: Includes danger penalties based on remaining time
- **State Management**: Tracks visited locations, remaining time, and current position
- **Validation Rules**: 
  - No revisiting non-Home locations (repeated visit vulnerability fixed)
  - Data collection time within bounds
  - Valid location indices only
  - No staying at current location

### Agent (`agent/dqn_agent_ablation.py`)
- **Intelligent Decision Making**: All smart components centralized in the agent
- **Components**: Action masking, safety mechanisms, constraint checking
- **Configurable**: Supports ablation studies with component enable/disable flags
- **Physical Feasibility**: Binds physical checks with action mask flag

### Key Features
1. **Action Masking**: Filters invalid actions before selection (155,385 rejections in experiments)
2. **Safety Mechanism**: 20% time threshold forced return home (26,232 interventions)
3. **Constraint Checking**: Real-time time constraint validation
4. **Risk-Aware Rewards**: Danger penalties for time-critical situations
5. **Uncertainty Handling**: Robust performance under data collection time variations

## Configuration
- **Environment Config**: `config/realword_8/config_8.json`
- **8 Locations + Home Base**: Real-world flight time matrices
- **Weather-dependent Flight Times**: Good/bad weather scenarios
- **Critical vs Low-Critical Locations**: HC=10x reward, LC=2x reward

## Reward System
```
R_data = 10 * T_data (HC locations) or 2 * T_data (LC locations) [only if unvisited]
Cost = -1 * (T_data + T_flight)
Danger_Penalty = 50 * (danger_ratio - 0.8)² [if danger_ratio > 0.8]
Total_Reward = R_data + Cost - Danger_Penalty
```

## Uncertainty Features
- **Data Collection Uncertainty**: Actual time = Planned time × (1.0-1.3 random factor)
- **Expected vs Actual Tracking**: Environment tracks both planned and actual times
- **Robustness Testing**: Components evaluated under realistic uncertainty conditions

## Ablation Study Models
1. **Vanilla DQN**: No intelligent components (baseline)
2. **DQN + Action Masking**: Filters invalid actions only
3. **DQN + Safety Mechanism**: 20% safety margin forced return home
4. **DQN + Constraint Checking**: Time constraint validation
5. **Complete Intelligent Agent**: All components enabled

## Usage
```bash
# Run ablation experiments
python3 ablation_study.py

# Run simple test
python3 simple_test.py
```

## Experimental Results (Under Uncertainty)

### Performance Comparison
| Model | Final Reward | Success Rate | Training Time | Key Statistics |
|-------|-------------|--------------|---------------|----------------|
| Vanilla DQN | -144,091.42 | 13% | 56.3s | 753 safety violations |
| Action Masking | **629.46** | **99%** | **24.0s** | 155,385 mask rejections |
| Safety Mechanism | -1,202.07 | 99% | 105.8s | 26,232 safety interventions |
| Constraint Checking | -279.11 | 100% | 201.2s | 1,903,766 constraint violations |
| Complete Agent | **629.46** | **99%** | **30.4s** | All components active |

### Key Insights
1. **Action Masking Dominance**: Most effective component under uncertainty
2. **Safety Mechanism Value**: Maintains 99% success rate despite negative rewards
3. **Uncertainty Impact**: Vanilla DQN performance severely degraded
4. **Component Synergy**: Complete agent matches action masking performance

## Technical Improvements
- **Fixed Repeated Visit Vulnerability**: Reward calculation before state update
- **Enhanced Error Handling**: Corrected UnboundLocalError in optimization loop
- **Uncertainty Integration**: Realistic data collection time variations
- **Risk-Aware Design**: Danger penalties for time-critical situations

## Results Directory
- **Ablation Results**: `ablation_results/` - Comprehensive analysis and visualizations
- **Configuration Files**: `config/` - Environment and experiment configurations
- **Agent Implementation**: `agent/` - DQN agent with intelligent components

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Gymnasium

## License
This project is licensed under the MIT License - see the LICENSE file for details.