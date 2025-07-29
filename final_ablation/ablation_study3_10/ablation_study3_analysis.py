import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_ablation_results():
    """Comprehensive analysis of ablation study results from ablation_study3."""
    
    # Define the experimental cases order (matches EXPERIMENT_CASES in ablation_study3.py)
    experiment_order = [
        # Group 1: Core Scaling Analysis (Stable Weather)
        {'name': '5_loc_stable', 'locations': 5, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05, 'group': 'Stable'},
        {'name': '8_loc_stable', 'locations': 8, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05, 'group': 'Stable'},
        {'name': '15_loc_stable', 'locations': 15, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05, 'group': 'Stable'},
        {'name': '20_loc_stable', 'locations': 20, 'weather_prob': 0.8, 'extreme_weather_prob': 0.05, 'group': 'Stable'},
        
        # Group 2: Robustness to Uncertainty (Unstable Weather)
        {'name': '5_loc_unstable', 'locations': 5, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15, 'group': 'Unstable'},
        {'name': '8_loc_unstable', 'locations': 8, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15, 'group': 'Unstable'},
        {'name': '15_loc_unstable', 'locations': 15, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15, 'group': 'Unstable'},
        {'name': '20_loc_unstable', 'locations': 20, 'weather_prob': 0.5, 'extreme_weather_prob': 0.15, 'group': 'Unstable'},
        
        # Group 3: Stress Test (Highly Unstable & Extreme Conditions)
        {'name': '15_loc_extreme', 'locations': 15, 'weather_prob': 0.4, 'extreme_weather_prob': 0.20, 'group': 'Extreme'},
        {'name': '20_loc_extreme', 'locations': 20, 'weather_prob': 0.4, 'extreme_weather_prob': 0.25, 'group': 'Extreme'},
    ]
    
    # Results directory
    results_dir = Path('/Users/captainzhang/Documents/1 Research/scheduling/Drone/Drone_route_plainning/ablation_results/final/ablation_study3_10')
    
    # Find all result directories
    result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(result_dirs)} result directories")
    
    # Collect all results
    all_results = []
    
    for i, result_dir in enumerate(result_dirs):
        if i >= len(experiment_order):
            print(f"Warning: More result directories than expected experiments")
            break
            
        experiment = experiment_order[i]
        csv_files = list(result_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV found in {result_dir}")
            continue
            
        csv_file = csv_files[0]  # Should be only one
        print(f"Processing: {experiment['name']} from {csv_file}")
        
        # Read the CSV
        df = pd.read_csv(csv_file)
        
        # Add experiment metadata to each row
        for _, row in df.iterrows():
            result_row = {
                'Experiment': experiment['name'],
                'Locations': experiment['locations'],
                'Weather_Prob': experiment['weather_prob'],
                'Extreme_Weather_Prob': experiment['extreme_weather_prob'],
                'Group': experiment['group'],
                'Model': row['Model'],
                'Action_Mask': row['Action Mask'],
                'Reward_Shaping': row['Reward Shaping'],
                'Eval_Base_Reward': row['Eval Base Reward'],
                'Eval_Total_Reward': row['Eval Total Reward'],
                'Eval_Success_Rate': row['Eval Success Rate (%)'],
                'Training_Final_Reward': row['Training Final Reward (Last 100)'],
                'Training_Success_Rate': row['Training Final Success (%) (Last 100)'],
                'Training_Time': row['Training Time (s)'],
                'Safety_Violations': row['Total Env Safety Violations'],
                'Constraint_Violations': row['Total Env Constraint Violations']
            }
            all_results.append(result_row)
    
    # Create comprehensive DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save the compiled results
    results_df.to_csv('ablation_study3_compiled_results.csv', index=False)
    print(f"Saved compiled results to ablation_study3_compiled_results.csv")
    
    # Generate summary table
    create_summary_table(results_df)
    
    # Generate analysis plots
    create_analysis_plots(results_df)
    
    return results_df

def create_summary_table(df):
    """Create a formatted summary table for the paper."""
    
    # Create pivot table for success rates
    success_pivot = df.pivot_table(
        values='Eval_Success_Rate', 
        index=['Locations', 'Group'], 
        columns='Model', 
        aggfunc='first'
    )
    
    # Create pivot table for base rewards  
    reward_pivot = df.pivot_table(
        values='Eval_Base_Reward', 
        index=['Locations', 'Group'], 
        columns='Model', 
        aggfunc='first'
    )
    
    # Create pivot table for safety violations
    safety_pivot = df.pivot_table(
        values='Safety_Violations', 
        index=['Locations', 'Group'], 
        columns='Model', 
        aggfunc='first'
    )
    
    print("\\n" + "="*80)
    print("ABLATION STUDY 3 RESULTS SUMMARY")
    print("="*80)
    
    print("\\n1. SUCCESS RATES (%)")
    print("-"*50)
    print(success_pivot.round(1))
    
    print("\\n2. EVALUATION BASE REWARDS")
    print("-"*50) 
    print(reward_pivot.round(1))
    
    print("\\n3. SAFETY VIOLATIONS (Total)")
    print("-"*50)
    print(safety_pivot.round(0).astype(int))
    
    # Create LaTeX table for paper
    create_latex_table(df)

def create_latex_table(df):
    """Create a LaTeX-formatted table for the paper."""
    
    # Prepare data for LaTeX table
    table_data = []
    
    experiments = df.groupby(['Locations', 'Group']).first().reset_index()
    
    for _, exp in experiments.iterrows():
        exp_data = df[(df['Locations'] == exp['Locations']) & (df['Group'] == exp['Group'])]
        
        row = {
            'scenario': f"{exp['Locations']} loc, {exp['Group']}",
            'vanilla_success': exp_data[exp_data['Model'] == '1. Vanilla DQN']['Eval_Success_Rate'].iloc[0],
            'vanilla_reward': exp_data[exp_data['Model'] == '1. Vanilla DQN']['Eval_Base_Reward'].iloc[0],
            'shaping_success': exp_data[exp_data['Model'] == '2. DQN + Shaping']['Eval_Success_Rate'].iloc[0],
            'shaping_reward': exp_data[exp_data['Model'] == '2. DQN + Shaping']['Eval_Base_Reward'].iloc[0],
            'masking_success': exp_data[exp_data['Model'] == '3. DQN + Action Masking']['Eval_Success_Rate'].iloc[0],
            'masking_reward': exp_data[exp_data['Model'] == '3. DQN + Action Masking']['Eval_Base_Reward'].iloc[0],
            'complete_success': exp_data[exp_data['Model'] == '4. Complete Agent']['Eval_Success_Rate'].iloc[0],
            'complete_reward': exp_data[exp_data['Model'] == '4. Complete Agent']['Eval_Base_Reward'].iloc[0],
        }
        table_data.append(row)
    
    # Generate LaTeX table
    latex_table = """
\\begin{table}[h]
\\centering
\\caption{Ablation Study Results: Performance Comparison Across Different Scenarios}
\\label{tab:ablation_results}
\\begin{tabular}{|l|cc|cc|cc|cc|}
\\hline
\\multirow{2}{*}{\\textbf{Scenario}} & \\multicolumn{2}{c|}{\\textbf{Vanilla DQN}} & \\multicolumn{2}{c|}{\\textbf{+ Reward Shaping}} & \\multicolumn{2}{c|}{\\textbf{+ Action Masking}} & \\multicolumn{2}{c|}{\\textbf{Complete Agent}} \\\\
& Success & Reward & Success & Reward & Success & Reward & Success & Reward \\\\
\\hline
"""
    
    for row in table_data:
        latex_table += f"{row['scenario']} & {row['vanilla_success']:.0f}\\% & {row['vanilla_reward']:.0f} & {row['shaping_success']:.0f}\\% & {row['shaping_reward']:.0f} & {row['masking_success']:.0f}\\% & {row['masking_reward']:.0f} & {row['complete_success']:.0f}\\% & {row['complete_reward']:.0f} \\\\\n"
    
    latex_table += """\\hline
\\end{tabular}
\\end{table}
"""
    
    with open('ablation_results_latex_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("\\nLaTeX table saved to 'ablation_results_latex_table.tex'")

def create_analysis_plots(df):
    """Create comprehensive analysis plots."""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ablation Study Analysis: Performance Across Different Scenarios', fontsize=16, fontweight='bold')
    
    # 1. Success Rate Comparison
    ax1 = axes[0, 0]
    success_data = df.pivot_table(values='Eval_Success_Rate', index='Locations', columns='Model', aggfunc='first')
    success_data.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Success Rate by Number of Locations', fontweight='bold')
    ax1.set_xlabel('Number of Locations')
    ax1.set_ylabel('Success Rate (%)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    
    # 2. Reward Performance Heatmap
    ax2 = axes[0, 1]
    reward_pivot = df.pivot_table(values='Eval_Base_Reward', index='Group', columns='Locations', aggfunc='mean')
    sns.heatmap(reward_pivot, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Base Reward'})
    ax2.set_title('Average Base Reward Heatmap', fontweight='bold')
    ax2.set_xlabel('Number of Locations')
    ax2.set_ylabel('Weather Condition')
    
    # 3. Safety Analysis
    ax3 = axes[1, 0]
    safety_data = df[df['Model'].isin(['1. Vanilla DQN', '4. Complete Agent'])]
    safety_comparison = safety_data.pivot_table(values='Safety_Violations', index='Locations', columns='Model', aggfunc='first')
    safety_comparison.plot(kind='bar', ax=ax3, color=['#e74c3c', '#2ecc71'])
    ax3.set_title('Safety Violations: Vanilla DQN vs Complete Agent', fontweight='bold')
    ax3.set_xlabel('Number of Locations')
    ax3.set_ylabel('Safety Violations')
    ax3.legend(['Vanilla DQN', 'Complete Agent'])
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    
    # 4. Training Efficiency
    ax4 = axes[1, 1]
    models_of_interest = ['1. Vanilla DQN', '3. DQN + Action Masking', '4. Complete Agent']
    training_data = df[df['Model'].isin(models_of_interest)]
    
    for model in models_of_interest:
        model_data = training_data[training_data['Model'] == model]
        ax4.scatter(model_data['Training_Time'], model_data['Eval_Success_Rate'], 
                   label=model.replace('DQN + ', ''), s=80, alpha=0.7)
    
    ax4.set_title('Training Efficiency: Time vs Success Rate', fontweight='bold')
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Success Rate (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_study3_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional focused plots
    create_scaling_analysis_plot(df)
    create_robustness_analysis_plot(df)

def create_scaling_analysis_plot(df):
    """Create a focused plot on performance scaling."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter stable conditions only
    stable_data = df[df['Group'] == 'Stable']
    
    # Success rate scaling
    for model in stable_data['Model'].unique():
        model_data = stable_data[stable_data['Model'] == model]
        ax1.plot(model_data['Locations'], model_data['Eval_Success_Rate'], 
                marker='o', linewidth=2.5, markersize=8, label=model.replace('DQN + ', ''))
    
    ax1.set_title('Performance Scaling in Stable Weather', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Number of Locations')
    ax1.set_ylabel('Success Rate (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Reward scaling
    for model in stable_data['Model'].unique():
        model_data = stable_data[stable_data['Model'] == model]
        ax2.plot(model_data['Locations'], model_data['Eval_Base_Reward'], 
                marker='s', linewidth=2.5, markersize=8, label=model.replace('DQN + ', ''))
    
    ax2.set_title('Reward Scaling in Stable Weather', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Number of Locations')  
    ax2.set_ylabel('Evaluation Base Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_robustness_analysis_plot(df):
    """Create analysis of robustness to weather conditions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Robustness Analysis: Performance Under Different Weather Conditions', fontsize=16, fontweight='bold')
    
    # Compare success rates across weather conditions for each location count
    location_counts = [5, 8, 15, 20]
    
    for i, loc_count in enumerate(location_counts):
        ax = axes[i//2, i%2]
        
        loc_data = df[df['Locations'] == loc_count]
        
        # Group by weather condition and model
        success_by_condition = loc_data.pivot_table(
            values='Eval_Success_Rate', 
            index='Group', 
            columns='Model', 
            aggfunc='first'
        )
        
        success_by_condition.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{loc_count} Locations: Weather Impact', fontweight='bold')
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Success Rate (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('weather_robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting comprehensive analysis of Ablation Study 3 results...")
    results_df = analyze_ablation_results()
    print(f"\\nAnalysis complete! Processed {len(results_df)} experiment results.")
    print("Generated files:")
    print("- ablation_study3_compiled_results.csv")
    print("- ablation_results_latex_table.tex") 
    print("- ablation_study3_comprehensive_analysis.png")
    print("- performance_scaling_analysis.png")
    print("- weather_robustness_analysis.png")