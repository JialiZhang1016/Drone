# Ablation Study Experiment Results

## Experiment Information
- **Date**: 2025_07_26_19:39:55
- **Locations**: 10
- **Episodes**: 2000
- **Config File**: config/config_10.json

## Files Description
- `experiment_config.json`: Complete experiment configuration including all parameters
- `config_10.json`: Original environment configuration file
- `ablation_analysis_10locations_V2.csv`: Detailed results table
- `ablation_analysis_10locations_V2.png`: Visualization charts

## Model Configurations
1. **Vanilla DQN**: No intelligent components
2. **DQN + Shaping**: Only reward shaping enabled
3. **DQN + Action Masking**: Action masking to filter invalid actions
4. **Complete Agent**: All intelligent components enabled

## Key Parameters
- **Training Episodes**: 2000
- **Epsilon Decay**: 0.995
- **Batch Size**: 64
- **Evaluation Window**: 100
- **Log Interval**: 500
