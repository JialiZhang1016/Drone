import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def safety_reward_shaping(safety_ratio, bonus_coefficient=1.0):
    """
    Risk-aware reward shaping function for drone route planning.
    
    Args:
        safety_ratio: T_return_home / T_remaining (higher values = more risky)
        bonus_coefficient: Scaling factor for the shaping bonus
    
    Returns:
        Shaping bonus that encourages safe behavior
    """
    return bonus_coefficient * np.maximum(0, 1 - safety_ratio)

def create_enhanced_visualization():
    """Create a clean visualization of the safety-aware reward shaping function."""
    
    # Use a clean, professional style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8
    })
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Safety ratios and shaping bonus
    safety_ratios = np.linspace(0, 2.0, 500)
    shaping_bonus = safety_reward_shaping(safety_ratios, bonus_coefficient=1.0)
    
    # Create background zones - only 2 zones as requested
    safe_zone = Rectangle((0, 0), 1.0, 1.1, alpha=0.2, color='green', label='Safe Zone')
    unsafe_zone = Rectangle((1.0, 0), 1.0, 1.1, alpha=0.2, color='red', label='Unsafe Zone')
    
    ax.add_patch(safe_zone)
    ax.add_patch(unsafe_zone)
    
    # Main shaping function curve
    ax.plot(safety_ratios, shaping_bonus, linewidth=4, color='#2E86AB', 
             label=r'$R_{\mathrm{shaping}} = \max(0, 1 - \frac{T_{\mathrm{home}}}{T_{\mathrm{rem}}})$', zorder=10)
    
    # Critical safety threshold
    ax.axvline(x=1.0, color='#A23B72', linestyle='--', linewidth=3, 
                label='Safety Threshold', zorder=5)
    
    # Add specific safety scenario points
    scenario_points = [(0.3, safety_reward_shaping(0.3)), (0.7, safety_reward_shaping(0.7)), 
                      (1.0, safety_reward_shaping(1.0)), (1.5, safety_reward_shaping(1.5))]
    scenario_labels = ['Very Safe\n(3× return time)', 'Moderately Safe\n(1.4× return time)', 
                      'Critical\n(Exact return time)', 'Unsafe\n(0.67× return time)']
    
    colors = ['#22A876', '#4CAF50', '#A23B72', '#C73E1D']
    
    for i, ((x, y), label) in enumerate(zip(scenario_points, scenario_labels)):
        ax.scatter(x, y, s=150, color=colors[i], zorder=15, edgecolor='white', linewidth=2)
        
        # Position annotations to avoid overlap
        va = 'bottom' if i < 2 else 'top'
        offset = 0.18 if i < 2 else -0.18
        ax.annotate(label, (x, y + offset), ha='center', va=va, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.4", fc='white', ec=colors[i], alpha=0.95))
    
    # Add zone text labels
    ax.text(0.5, 0.8, 'Safe Zone', fontsize=14, ha='center', va='center', 
            fontweight='bold', color='darkgreen', alpha=0.8)
    ax.text(1.5, 0.8, 'Unsafe Zone', fontsize=14, ha='center', va='center', 
            fontweight='bold', color='darkred', alpha=0.8)
    
    ax.set_title('Safety-Aware Reward Shaping Function', fontsize=16, fontweight='bold', pad=25)
    ax.set_xlabel(r'Safety Ratio ($\frac{T_{\mathrm{return\_home}}}{T_{\mathrm{remaining}}}$)', fontsize=14)
    ax.set_ylabel('Shaping Bonus', fontsize=14)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('safety_reward_shaping_function.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Create the enhanced visualization
    create_enhanced_visualization()
    
    # Print key insights
    print("Key Benefits of Safety-Aware Reward Shaping:")
    print("1. Encourages proactive safety behavior during exploration")
    print("2. Provides continuous guidance rather than sparse safety penalties")
    print("3. Accelerates learning of safe policies")
    print("4. Maintains task performance while improving safety margins")