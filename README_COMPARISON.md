# Unified Algorithm Comparison Framework

A comprehensive framework for comparing reinforcement learning and traditional algorithms for drone route planning problems.

## 🚀 Features

### **Algorithm Support**
- **Deep RL**: DQN variants, Double DQN, Dueling DQN
- **Modern RL**: PPO, A2C, SAC (via Stable Baselines3)
- **Heuristic**: Greedy, Rule-based algorithms
- **Path Planning**: Shortest path (Dijkstra-based)
- **Optimization**: Genetic Algorithm
- **Baseline**: Random, Priority-based

### **Comparison Capabilities**
- ✅ **Fair Comparison**: Unified training conditions and evaluation
- ✅ **Statistical Testing**: Friedman tests, pairwise comparisons, effect sizes
- ✅ **Scalability Analysis**: Performance across environment complexity
- ✅ **Robustness Evaluation**: Performance across different conditions
- ✅ **Efficiency Metrics**: Training time, computational resources
- ✅ **Reproducibility**: Fixed seeds, standardized configurations

### **Visualization & Analysis**
- 📊 Performance ranking with confidence intervals
- 📈 Scalability curves across environment sizes
- 🔥 Robustness heatmaps
- 📉 Statistical significance testing
- ⚡ Computational efficiency analysis

## 📁 File Structure

```
📦 Unified Comparison Framework
├── 🎯 unified_comparison.py          # Main comparison framework
├── 🔧 comparison_utils.py            # Utility functions
├── 📊 advanced_analysis.py           # Advanced statistical analysis
├── 🚀 run_full_comparison.py         # Quick start script
├── ⚙️ comparison_config.yaml         # Configuration file
├── 🤖 agent/
│   ├── dqn_agent_ablation.py         # Main DQN implementation
│   ├── heuristic_agents.py           # Greedy & Rule-based
│   └── advanced_agents.py            # Additional baselines
└── 📝 README_COMPARISON.md           # This file
```

## 🏃‍♂️ Quick Start

### **1. Basic Usage**
```bash
# Run default comparison (all algorithms, 3 seeds)
python run_full_comparison.py

# Quick test with reduced parameters
python run_full_comparison.py --quick

# Specific algorithms only
python run_full_comparison.py --algorithms dqn_full,greedy,random

# Specific environments
python run_full_comparison.py --environments config_5.json,config_8.json
```

### **2. Advanced Usage**
```bash
# Custom configuration
python run_full_comparison.py --config my_config.yaml

# Different seeds for statistical power
python run_full_comparison.py --seeds 42,123,456,789,999

# Dry run to check configuration
python run_full_comparison.py --dry-run
```

### **3. Direct Framework Usage**
```python
from unified_comparison import UnifiedComparisonFramework

# Initialize framework
framework = UnifiedComparisonFramework('comparison_config.yaml')

# Run comparison
results_df = framework.run_full_comparison()

# Generate analysis
output_dir = framework.save_results(results_df)
framework.generate_analysis(results_df, output_dir)
```

## ⚙️ Configuration

### **Algorithm Configuration**
```yaml
algorithms:
  dqn_full:
    type: "dqn"
    components:
      action_mask: true
      safety: true
      constraint: true
      reward_shaping: true
  
  greedy:
    type: "heuristic"
    class: "GreedyAgent"
  
  double_dqn:
    type: "advanced_rl"
    class: "DoubleDQNAgent"
```

### **Training Parameters**
```yaml
training:
  episodes: 2000
  batch_size: 64
  learning_rate: 0.001
  epsilon_start: 1.0
  epsilon_end: 0.02
  epsilon_decay: 0.995
```

### **Evaluation Settings**
```yaml
evaluation:
  episodes: 500
  final_eval_episodes: 100
  seeds: [42, 123, 456]
  metrics:
    - "cumulative_reward"
    - "success_rate"
    - "episode_length"
    - "training_time"
```

## 📊 Results and Analysis

### **Generated Files**
After running a comparison, the framework generates:

#### **Data Files**
- `comparison_results.csv` - Raw experimental results
- `algorithm_ranking.csv` - Performance rankings with confidence intervals
- `statistical_analysis.csv` - Pairwise statistical test results
- `scalability_data.csv` - Performance vs environment complexity
- `robustness_analysis.csv` - Robustness metrics across conditions

#### **Visualizations**
- `algorithm_ranking.png` - Performance ranking charts
- `scalability_analysis.png` - Scalability curves
- `robustness_heatmap.png` - Performance heatmap
- `performance_distributions.png` - Statistical distributions
- `computational_efficiency.png` - Training efficiency analysis

#### **Reports**
- `comprehensive_report.md` - Executive summary with recommendations
- `comparison_config.json` - Configuration backup

### **Statistical Analysis**
The framework performs:
- **Friedman Tests**: Non-parametric ANOVA for each environment
- **Pairwise Comparisons**: Wilcoxon signed-rank tests between algorithms
- **Effect Size Calculation**: Cohen's d for practical significance
- **Multiple Testing Correction**: Bonferroni correction for p-values
- **Confidence Intervals**: Bootstrap confidence intervals for metrics

## 🔬 Algorithm Details

### **Deep Reinforcement Learning**
| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **DQN Vanilla** | Basic Deep Q-Network | Standard RL baseline |
| **DQN + Reward Shaping** | DQN with reward engineering | Improved learning signal |
| **DQN + Action Masking** | DQN with invalid action filtering | Constraint handling |
| **Complete DQN** | All intelligent components | Full framework |
| **Double DQN** | Reduces overestimation bias | Improved Q-value estimates |
| **Dueling DQN** | Separate value/advantage streams | Better value function learning |

### **Heuristic & Traditional**
| Algorithm | Description | Computational Cost |
|-----------|-------------|-------------------|
| **Greedy** | Utility maximization | O(n) per decision |
| **Rule-based** | Priority-driven decisions | O(1) per decision |
| **Shortest Path** | Dijkstra-based planning | O(n²) preprocessing |
| **Genetic Algorithm** | Evolutionary optimization | O(population × generations) |
| **Priority-based** | Criticality + proximity scoring | O(n) per decision |

## 📈 Performance Metrics

### **Primary Metrics**
- **Cumulative Reward**: Total reward achieved
- **Success Rate**: Percentage of successful episodes (safe return)
- **Episode Length**: Average steps to completion
- **Training Time**: Computational cost (RL algorithms only)

### **Advanced Metrics**
- **Efficiency**: Reward per step, reward per training second
- **Robustness**: Performance variance across environments
- **Scalability**: Performance degradation with complexity
- **Constraint Violations**: Safety and feasibility violations

## 🛠️ Customization

### **Adding New Algorithms**
1. **Implement Agent Class**:
```python
class MyNewAgent:
    def __init__(self, env):
        self.env = env
        # Initialize agent
    
    def select_action(self, state, epsilon=0.0):
        # Return action index
        pass
```

2. **Add to Configuration**:
```yaml
algorithms:
  my_algorithm:
    type: "custom"
    class: "MyNewAgent"
```

3. **Update Framework**:
```python
elif algo_type == 'custom':
    if agent_class == 'MyNewAgent':
        return MyNewAgent(env)
```

### **Custom Metrics**
Add new metrics by modifying the `evaluate_agent` method:
```python
def evaluate_agent(self, agent, env, algorithm_name, seed):
    # ... existing code ...
    
    # Add custom metric
    custom_metric = self.calculate_custom_metric(agent, env)
    
    return {
        # ... existing metrics ...
        "custom_metric": custom_metric
    }
```

## 🔍 Troubleshooting

### **Common Issues**
1. **Missing Dependencies**: Install required packages
```bash
pip install torch pandas matplotlib seaborn scipy scikit-posthocs pyyaml
```

2. **GPU Memory Issues**: Reduce batch size or use CPU
```yaml
training:
  batch_size: 32  # Reduce from 64
```

3. **Long Training Times**: Enable quick mode
```bash
python run_full_comparison.py --quick
```

### **Performance Optimization**
- **Parallel Evaluation**: Set `max_workers` in config
- **GPU Usage**: Ensure CUDA is available for RL algorithms
- **Memory Management**: Monitor RAM usage for large experiments

## 📚 References

### **Algorithms**
- **DQN**: Mnih et al. (2015) - Human-level control through deep reinforcement learning
- **Double DQN**: van Hasselt et al. (2016) - Deep Reinforcement Learning with Double Q-learning
- **Dueling DQN**: Wang et al. (2016) - Dueling Network Architectures for Deep Reinforcement Learning

### **Statistical Methods**
- **Friedman Test**: Non-parametric repeated measures ANOVA
- **Wilcoxon Test**: Non-parametric pairwise comparison
- **Cohen's d**: Effect size measure for practical significance

## 📄 License

This comparison framework is part of the drone route planning research project.

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-algorithm`
3. **Add your algorithm implementation**
4. **Update configuration and documentation**
5. **Submit a pull request**

---

**For questions or issues, please refer to the troubleshooting section or create an issue in the repository.**