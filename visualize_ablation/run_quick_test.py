import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

def generate_mock_data():
    """
    Generates a mock DataFrame simulating the results of a full experiment suite.
    This data is designed to be realistic for visualization purposes.
    """
    scenarios = [
        {'locations': 8, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05},
        {'locations': 8, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
        {'locations': 15, 'weather_prob': 0.7, 'extreme_weather_prob': 0.10},
        {'locations': 15, 'weather_prob': 0.4, 'extreme_weather_prob': 0.20},
        {'locations': 20, 'weather_prob': 0.7, 'extreme_weather_prob': 0.10},
        {'locations': 20, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15},
    ]

    model_configs = [
        '1. Vanilla DQN',
        '2. DQN + Shaping',
        '3. DQN + Action Masking',
        '4. Complete Agent'
    ]
    
    all_results = []

    for scenario in scenarios:
        # Base success rate decreases with more locations and worse weather
        base_success = 1.0 - (scenario['locations'] / 30.0) - (1 - scenario['weather_prob']) / 2.0
        
        for i, model_name in enumerate(model_configs):
            # Model performance relative to base (Complete Agent is best)
            model_factor = 0.65 + (i * 0.1) 
            
            # Add noise
            noise = np.random.uniform(-0.05, 0.05)
            
            # Calculate final metrics
            success_rate = max(0, min(1, base_success * model_factor + noise))
            avg_reward = (success_rate * (scenario['locations'] * 50)) - ((1 - success_rate) * 500)
            training_time = 1200 + (scenario['locations'] * 100) + np.random.uniform(-100, 100)

            all_results.append({
                'locations': scenario['locations'],
                'weather_prob': scenario['weather_prob'],
                'extreme_weather_prob': scenario['extreme_weather_prob'],
                'model_name': model_name,
                'eval_success_rate': success_rate,
                'eval_avg_reward': avg_reward,
                'training_time': training_time
            })
            
    return pd.DataFrame(all_results)

def create_master_summary_table(df):
    """
    Creates and prints a master summary table showing the best agent's performance
    across all scenarios.
    """
    print("="*80)
    print("Master Performance Summary Table (Showing 'Complete Agent' Performance)")
    print("="*80)

    # Filter for the best model
    summary_df = df[df['model_name'] == '4. Complete Agent'].copy()

    # Create a scenario description column
    summary_df['Scenario'] = summary_df.apply(
        lambda row: f"Loc: {row['locations']}, WP: {row['weather_prob']}, EWP: {row['extreme_weather_prob']}",
        axis=1
    )

    # Format for display
    summary_df['Eval Success Rate (%)'] = (summary_df['eval_success_rate'] * 100).map('{:.1f}'.format)
    summary_df['Eval Avg Reward'] = summary_df['eval_avg_reward'].map('{:.0f}'.format)
    summary_df['Training Time (s)'] = summary_df['training_time'].map('{:.0f}'.format)

    # Select and reorder columns for the final table
    display_table = summary_df[['Scenario', 'Eval Success Rate (%)', 'Eval Avg Reward', 'Training Time (s)']]
    
    print(display_table.to_string(index=False))
    print("="*80 + "\n")


def plot_performance_scaling(df):
    """
    Generates a line plot showing how agent performance scales with the number of locations.
    """
    print("Generating plot: Performance Scaling vs. Number of Locations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter for the models and scenarios you want to compare
    plot_df = df[df['model_name'].isin(['1. Vanilla DQN', '4. Complete Agent'])]
    # Use only the stable weather scenarios for a clearer comparison
    plot_df = plot_df[plot_df['weather_prob'] >= 0.7]

    sns.lineplot(
        data=plot_df,
        x='locations',
        y='eval_success_rate',
        hue='model_name',
        style='model_name',
        markers=True,
        dashes=False,
        ax=ax,
        linewidth=2.5,
        markersize=8
    )

    ax.set_title('Performance Scaling with Problem Size', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Locations', fontsize=12)
    ax.set_ylabel('Evaluation Success Rate', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    ax.set_xticks(df['locations'].unique())
    ax.legend(title='Agent Model')
    
    plt.tight_layout()
    plt.savefig('scaling_vs_locations.png', dpi=300)
    print("Saved 'scaling_vs_locations.png'\n")
    plt.close()


def plot_weather_robustness(df):
    """
    Generates a grouped bar chart showing agent robustness to weather conditions.
    """
    print("Generating plot: Robustness to Weather Conditions...")
    plt.style.use('seaborn-v0_8-talk')
    
    # Analyze only the best agent
    plot_df = df[df['model_name'] == '4. Complete Agent'].copy()

    # Categorize weather
    plot_df['Weather Condition'] = plot_df['weather_prob'].apply(lambda x: 'Stable' if x >= 0.7 else 'Unstable')

    g = sns.catplot(
        data=plot_df,
        x='locations',
        y='eval_success_rate',
        hue='Weather Condition',
        kind='bar',
        palette={'Stable': 'skyblue', 'Unstable': 'salmon'},
        edgecolor='black'
    )

    g.fig.suptitle('Agent Robustness to Weather Conditions', y=1.03, fontsize=16, fontweight='bold')
    g.set_axis_labels('Number of Locations', 'Evaluation Success Rate')
    
    # Format y-axis as percentage
    ax = g.axes[0,0]
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    plt.tight_layout()
    plt.savefig('robustness_vs_weather.png', dpi=300)
    print("Saved 'robustness_vs_weather.png'\n")
    plt.close()


def plot_performance_heatmap(df):
    """
    Generates a heatmap showing the best agent's performance across all scenarios.
    """
    print("Generating plot: Performance Heatmap...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter for the best model
    plot_df = df[df['model_name'] == '4. Complete Agent'].copy()
    
    # Create a weather difficulty metric
    plot_df['Weather Difficulty'] = (
        (1 - plot_df['weather_prob']) + plot_df['extreme_weather_prob']
    ).round(2)

    # Pivot the data for the heatmap
    heatmap_data = plot_df.pivot_table(
        index='locations',
        columns='Weather Difficulty',
        values='eval_success_rate'
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1%",
        linewidths=.5,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Evaluation Success Rate'}
    )

    ax.set_title('Complete Agent Performance Across Scenarios', fontsize=16, fontweight='bold')
    ax.set_xlabel('Weather Difficulty ((1-P_good) + P_extreme)', fontsize=12)
    ax.set_ylabel('Number of Locations', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300)
    print("Saved 'performance_heatmap.png'\n")
    plt.close()


if __name__ == '__main__':
    # 1. Generate the simulated data
    results_df = generate_mock_data()

    # 2. Create and print the master summary table
    create_master_summary_table(results_df)

    # 3. Create and save the recommended cross-scenario graphs
    plot_performance_scaling(results_df)
    plot_weather_robustness(results_df)
    plot_performance_heatmap(results_df)