#!/usr/bin/env python3
"""
Quick Start Script for Unified Comparison Framework

This script provides a convenient way to run the full comparison study
with different configurations and settings.

Usage Examples:
    # Run default comparison
    python run_full_comparison.py
    
    # Run with specific algorithms
    python run_full_comparison.py --algorithms dqn_full,greedy,random
    
    # Run with specific environments
    python run_full_comparison.py --environments config_5.json,config_8.json
    
    # Run with custom seeds
    python run_full_comparison.py --seeds 42,123,456,789,999
    
    # Quick test run
    python run_full_comparison.py --quick
"""

import argparse
import sys
import os
from datetime import datetime
import yaml

def create_custom_config(args):
    """Create configuration based on command line arguments"""
    
    # Default configuration
    config = {
        "algorithms": {
            "dqn_vanilla": {"type": "dqn", "components": {"action_mask": False, "safety": False, "constraint": False, "reward_shaping": False}},
            "dqn_reward": {"type": "dqn", "components": {"action_mask": False, "safety": False, "constraint": False, "reward_shaping": True}},
            "dqn_mask": {"type": "dqn", "components": {"action_mask": True, "safety": False, "constraint": False, "reward_shaping": False}},
            "dqn_full": {"type": "dqn", "components": {"action_mask": True, "safety": True, "constraint": True, "reward_shaping": True}},
            "double_dqn": {"type": "advanced_rl", "class": "DoubleDQNAgent"},
            "dueling_dqn": {"type": "advanced_rl", "class": "DuelingDQNAgent"},
            "greedy": {"type": "heuristic", "class": "GreedyAgent"},
            "rule_based": {"type": "heuristic", "class": "RuleBasedAgent"},
            "shortest_path": {"type": "path_planning", "class": "ShortestPathAgent"},
            "genetic_algorithm": {"type": "optimization", "class": "GeneticAlgorithmAgent"},
            "priority_based": {"type": "priority", "class": "PriorityAgent"},
            "random": {"type": "baseline", "class": "RandomAgent"}
        },
        "environments": ["config_5.json", "config_8.json", "config_10.json"],
        "training": {
            "episodes": 2000,
            "batch_size": 64,
            "learning_rate": 0.001,
            "epsilon_start": 1.0,
            "epsilon_end": 0.02,
            "epsilon_decay": 0.995,
            "target_update_freq": 100,
            "memory_capacity": 10000,
            "gamma": 0.99
        },
        "evaluation": {
            "episodes": 500,
            "final_eval_episodes": 100,
            "seeds": [42, 123, 456],
            "metrics": ["cumulative_reward", "success_rate", "episode_length", "training_time", "constraint_violations"]
        },
        "output": {
            "results_dir": "unified_comparison_results",
            "save_models": True,
            "generate_plots": True,
            "statistical_tests": True
        }
    }
    
    # Apply command line overrides
    if args.algorithms:
        selected_algorithms = args.algorithms.split(',')
        config["algorithms"] = {k: v for k, v in config["algorithms"].items() 
                               if k in selected_algorithms}
    
    if args.environments:
        config["environments"] = args.environments.split(',')
    
    if args.seeds:
        config["evaluation"]["seeds"] = [int(s) for s in args.seeds.split(',')]
    
    if args.quick:
        # Quick test configuration
        config["training"]["episodes"] = 100
        config["evaluation"]["episodes"] = 50
        config["evaluation"]["final_eval_episodes"] = 10
        config["environments"] = ["config_5.json"]
        # Keep only a few algorithms for quick test
        quick_algorithms = ["dqn_vanilla", "dqn_full", "greedy", "random"]
        config["algorithms"] = {k: v for k, v in config["algorithms"].items() 
                               if k in quick_algorithms}
    
    if args.episodes:
        config["training"]["episodes"] = args.episodes
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Run Unified Algorithm Comparison")
    
    # Algorithm selection
    parser.add_argument('--algorithms', type=str,
                       help='Comma-separated list of algorithms to test')
    
    # Environment selection
    parser.add_argument('--environments', type=str,
                       help='Comma-separated list of environment configs')
    
    # Seeds
    parser.add_argument('--seeds', type=str,
                       help='Comma-separated list of random seeds')
    
    # Training parameters
    parser.add_argument('--episodes', type=int,
                       help='Number of training episodes')
    
    # Quick test mode
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with reduced parameters')
    
    # Output directory
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='comparison_config.yaml',
                       help='Configuration file path')
    
    # Dry run
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without running')
    
    args = parser.parse_args()
    
    print("🚀 Unified Algorithm Comparison Framework")
    print("=" * 50)
    
    # Create configuration
    if os.path.exists(args.config) and not any([args.algorithms, args.environments, 
                                               args.seeds, args.quick, args.episodes]):
        print(f"📋 Using configuration file: {args.config}")
        config_file = args.config
    else:
        print("📋 Creating custom configuration from command line arguments")
        config = create_custom_config(args)
        
        # Save temporary config file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = f"temp_config_{timestamp}.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"📋 Temporary configuration saved: {config_file}")
    
    # Show configuration summary
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n📊 Experiment Configuration:")
    print(f"   Algorithms: {list(config['algorithms'].keys())}")
    print(f"   Environments: {config['environments']}")
    print(f"   Seeds: {config['evaluation']['seeds']}")
    print(f"   Training Episodes: {config['training']['episodes']}")
    print(f"   Evaluation Episodes: {config['evaluation']['final_eval_episodes']}")
    
    total_experiments = (len(config['algorithms']) * 
                        len(config['environments']) * 
                        len(config['evaluation']['seeds']))
    print(f"   Total Experiments: {total_experiments}")
    
    if args.dry_run:
        print("\n🔍 Dry run completed. Configuration looks good!")
        if config_file.startswith('temp_config_'):
            os.remove(config_file)
        return
    
    # Auto-confirm for full experiments 
    if not args.quick and total_experiments > 50:
        print(f"\n🚀 Starting {total_experiments} experiments (auto-confirmed)...")
    
    # Run the comparison
    try:
        from unified_comparison import UnifiedComparisonFramework
        
        print(f"\n🏃 Starting comparison framework...")
        framework = UnifiedComparisonFramework(config_file)
        
        # Run comparison
        results_df = framework.run_full_comparison()
        
        # Generate analysis
        output_dir = framework.save_results(results_df)
        framework.generate_analysis(results_df, output_dir)
        
        print(f"\n🎉 Comparison completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        
        # Show quick summary
        print(f"\n📊 Quick Results Summary:")
        summary = results_df.groupby('algorithm')['cumulative_reward'].agg(['mean', 'std']).round(2)
        print(summary.sort_values('mean', ascending=False))
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up temporary config file
        if config_file.startswith('temp_config_'):
            try:
                os.remove(config_file)
                print(f"🧹 Cleaned up temporary config: {config_file}")
            except:
                pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())