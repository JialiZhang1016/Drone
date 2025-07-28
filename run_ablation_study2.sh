#!/bin/bash
#SBATCH --job-name=run_ablation_study2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:V100-SXM2-32GB:2
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=jztk7@mst.edu
#SBATCH --output=runs/output/Mill2-%j.out
#SBATCH --error=runs/output/Mill2-%j.err

# Load necessary modules
module load anaconda

# Activate your conda environment (adjust the environment name as needed)
# conda activate your_env_name

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Print GPU information
echo "GPU information:"
nvidia-smi

# Change to the correct directory
cd /home/jztk7/Desktop/Drone

# Print Python and package versions for reproducibility
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

# Run the enhanced ablation study with model saving and evaluation
echo "Starting enhanced ablation study with model saving..."
echo "Running ablation_study2.py"

# Option 1: Run with default config (config_5.json)
python ablation_study2.py

# Option 2: Run with specific configuration (uncomment one of these lines if you want to run a specific config)
# python ablation_study2.py config/config_5.json
# python ablation_study2.py config/config_8.json
# python ablation_study2.py config/config_10.json

# Option 3: Run multiple configurations sequentially (uncomment to run all)
# python ablation_study2.py config/config_5.json config/config_8.json config/config_10.json

echo "Enhanced ablation study completed at: $(date)"

# List the generated results
echo "Generated results:"
ls -la ablation_results/

# Show saved models
echo "Saved models in latest results directory:"
latest_dir=$(ls -t ablation_results/ | head -1)
if [ -n "$latest_dir" ]; then
    echo "Latest results directory: $latest_dir"
    ls -la "ablation_results/$latest_dir/"*.pth 2>/dev/null || echo "No model files found"
fi