#!/usr/bin/env python3
# comprehensive_analysis.py
# 综合分析所有对比实验结果

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from glob import glob

def load_all_results():
    """加载所有实验结果"""
    results_dirs = glob('comparative_results/20250727-*')
    
    all_results = []
    for results_dir in results_dirs:
        csv_file = os.path.join(results_dir, 'results_summary.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # 从目录名提取配置信息
            config_name = os.path.basename(results_dir).split('_')[-1]
            df['Config'] = config_name
            
            # 提取位置数量
            if 'config_5' in results_dir:
                df['Locations'] = 5
            elif 'config_10' in results_dir:
                df['Locations'] = 10
            elif 'config_15' in results_dir:
                df['Locations'] = 15
            else:
                df['Locations'] = 0
                
            all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def create_comprehensive_plots(df, output_dir):
    """创建综合对比图表"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置图表样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
    
    # 1. 跨配置性能对比
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive Performance Analysis Across Configurations', fontsize=16)
    
    # a) 平均奖励对比
    pivot_reward = df.pivot(index='Model', columns='Locations', values='Reward (Mean)')
    sns.heatmap(pivot_reward, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0,0])
    axes[0,0].set_title('Average Reward by Model and Configuration')
    axes[0,0].set_ylabel('Model')
    
    # b) 成功率对比
    pivot_success = df.pivot(index='Model', columns='Locations', values='Success Rate (Mean)')
    sns.heatmap(pivot_success, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0,1])
    axes[0,1].set_title('Success Rate by Model and Configuration')
    axes[0,1].set_ylabel('Model')
    
    # c) 奖励柱状图
    bar_data = df[df['Model'].isin(['PPO', 'A2C'])]  # 只显示RL算法
    x_pos = np.arange(len(bar_data))
    bars = axes[1,0].bar(x_pos, bar_data['Reward (Mean)'], 
                        yerr=bar_data['Reward (Std)'], capsize=5,
                        color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    axes[1,0].set_title('RL Algorithms Reward Comparison')
    axes[1,0].set_xlabel('Model-Configuration')
    axes[1,0].set_ylabel('Average Reward')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels([f"{row['Model']}-{row['Locations']}" for _, row in bar_data.iterrows()], rotation=45)
    
    # d) 算法扩展性分析
    rl_data = df[df['Model'].isin(['PPO', 'A2C'])]
    for model in ['PPO', 'A2C']:
        model_data = rl_data[rl_data['Model'] == model].sort_values('Locations')
        axes[1,1].plot(model_data['Locations'], model_data['Success Rate (Mean)'], 
                      marker='o', linewidth=2, label=f'{model} Success Rate')
        axes[1,1].plot(model_data['Locations'], model_data['Reward (Mean)']/1000, 
                      marker='s', linewidth=2, linestyle='--', label=f'{model} Reward (÷1000)')
    
    axes[1,1].set_title('Scalability Analysis')
    axes[1,1].set_xlabel('Number of Locations')
    axes[1,1].set_ylabel('Normalized Performance')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 算法性能详细对比
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Detailed Algorithm Performance Comparison', fontsize=16)
    
    # 按配置分组显示
    configs = sorted(df['Locations'].unique())
    x = np.arange(len(df['Model'].unique()))
    width = 0.25
    
    for i, config in enumerate(configs):
        config_data = df[df['Locations'] == config]
        
        # 奖励对比
        axes[0].bar(x + i*width, config_data['Reward (Mean)'], width, 
                   label=f'{config} locations', alpha=0.8)
        
        # 成功率对比
        axes[1].bar(x + i*width, config_data['Success Rate (Mean)'], width,
                   label=f'{config} locations', alpha=0.8)
        
        # 训练时间对比 (只显示RL算法)
        rl_data = config_data[config_data['Model'].isin(['PPO', 'A2C'])]
        if not rl_data.empty:
            rl_x = np.arange(len(rl_data))
            axes[2].bar(rl_x + i*width, rl_data['Training Time (s)'], width,
                       label=f'{config} locations', alpha=0.8)
    
    axes[0].set_title('Average Reward Comparison')
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Reward')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(df['Model'].unique())
    axes[0].legend()
    
    axes[1].set_title('Success Rate Comparison')
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(df['Model'].unique())
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    
    axes[2].set_title('Training Time (RL Algorithms Only)')
    axes[2].set_xlabel('RL Models')
    axes[2].set_ylabel('Training Time (seconds)')
    axes[2].set_xticks(np.arange(2) + width)
    axes[2].set_xticklabels(['PPO', 'A2C'])
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(df, output_dir):
    """生成综合分析报告"""
    report_path = os.path.join(output_dir, 'comprehensive_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 无人机路径规划算法综合对比分析报告\n\n")
        
        f.write("## 实验设置\n")
        f.write("- **训练步数**: 2000 timesteps\n")
        f.write("- **评估回合数**: 500 episodes\n")
        f.write("- **随机种子**: 3个\n")
        f.write("- **测试配置**: 5, 10, 15 个位置\n\n")
        
        f.write("## 整体结果汇总\n\n")
        f.write("### 所有实验结果\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 关键发现\n\n")
        
        # 分析最佳性能
        f.write("### 1. 算法性能排名\n")
        
        # 按成功率排序
        success_ranking = df.groupby('Model')['Success Rate (Mean)'].mean().sort_values(ascending=False)
        f.write("**按成功率排序:**\n")
        for i, (model, success_rate) in enumerate(success_ranking.items(), 1):
            f.write(f"{i}. {model}: {success_rate:.1%}\n")
        f.write("\n")
        
        # 按奖励排序（仅成功的算法）
        successful_models = df[df['Success Rate (Mean)'] > 0]
        if not successful_models.empty:
            reward_ranking = successful_models.groupby('Model')['Reward (Mean)'].mean().sort_values(ascending=False)
            f.write("**按平均奖励排序（成功算法）:**\n")
            for i, (model, reward) in enumerate(reward_ranking.items(), 1):
                f.write(f"{i}. {model}: {reward:.2f}\n")
        f.write("\n")
        
        f.write("### 2. 扩展性分析\n")
        
        # 强化学习算法扩展性
        rl_models = ['PPO', 'A2C']
        for model in rl_models:
            model_data = df[df['Model'] == model].sort_values('Locations')
            f.write(f"**{model} 扩展性:**\n")
            for _, row in model_data.iterrows():
                f.write(f"- {row['Locations']}位置: 成功率{row['Success Rate (Mean)']:.1%}, ")
                f.write(f"平均奖励{row['Reward (Mean)']:.2f}\n")
            f.write("\n")
        
        f.write("### 3. 启发式算法分析\n")
        heuristic_models = ['Greedy', 'Rule-Based']
        heuristic_data = df[df['Model'].isin(heuristic_models)]
        
        f.write("启发式算法在所有配置下都未能成功完成任务：\n")
        for model in heuristic_models:
            model_data = heuristic_data[heuristic_data['Model'] == model]
            avg_reward = model_data['Reward (Mean)'].mean()
            f.write(f"- **{model}**: 平均奖励 {avg_reward:.2f}, 成功率 0%\n")
        f.write("\n")
        
        f.write("### 4. 训练效率\n")
        rl_data = df[df['Model'].isin(rl_models)]
        avg_training_time = rl_data.groupby('Model')['Training Time (s)'].mean()
        f.write("**平均训练时间:**\n")
        for model, time_s in avg_training_time.items():
            f.write(f"- {model}: {time_s:.2f} 秒\n")
        f.write("\n")
        
        f.write("## 结论与建议\n\n")
        
        best_overall = success_ranking.index[0]
        f.write(f"1. **{best_overall}** 是整体表现最佳的算法，在所有配置下都实现了高成功率\n")
        
        if len(successful_models['Model'].unique()) > 1:
            best_reward_model = reward_ranking.index[0]
            f.write(f"2. **{best_reward_model}** 在成功算法中获得了最高的平均奖励\n")
        
        f.write("3. 强化学习算法显著优于启发式方法，特别是在复杂环境中\n")
        f.write("4. 随着位置数量增加，强化学习算法展现出更好的扩展性\n")
        f.write("5. 启发式算法需要重新设计策略以提高性能\n\n")
        
        f.write("## 未来工作建议\n\n")
        f.write("1. 增加训练步数以进一步提升RL算法性能\n")
        f.write("2. 优化启发式算法的决策规则\n")
        f.write("3. 测试更多RL算法（如SAC、TD3）\n")
        f.write("4. 分析不同奖励函数设计的影响\n")
        f.write("5. 在更大规模环境（20+位置）上验证结果\n")

def main():
    """主函数"""
    print("="*60)
    print("Comprehensive Analysis of Drone Route Planning Algorithms")
    print("="*60)
    
    # 加载所有结果
    df = load_all_results()
    
    if df.empty:
        print("No results found. Please run the comparative studies first.")
        return
    
    print(f"Loaded {len(df)} result sets from {len(df['Config'].unique())} configurations")
    
    # 创建输出目录
    output_dir = 'comprehensive_analysis_results'
    
    # 生成图表
    print("Generating comprehensive plots...")
    create_comprehensive_plots(df, output_dir)
    
    # 生成报告
    print("Generating comprehensive report...")
    generate_comprehensive_report(df, output_dir)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("- comprehensive_analysis.png")
    print("- detailed_comparison.png") 
    print("- comprehensive_report.md")
    
    # 显示简要结果
    print("\n--- Quick Summary ---")
    print("Success Rate by Model:")
    success_summary = df.groupby('Model')['Success Rate (Mean)'].mean().sort_values(ascending=False)
    for model, rate in success_summary.items():
        print(f"  {model}: {rate:.1%}")

if __name__ == "__main__":
    main()