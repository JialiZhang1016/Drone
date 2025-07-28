#!/bin/bash
#SBATCH --job-name=run_ablation_study2_multi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:V100-SXM2-32GB:2
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=jztk7@mst.edu
#SBATCH --output=runs/output/Mill2_multi_gpu-%j.out
#SBATCH --error=runs/output/Mill2_multi_gpu-%j.err

# Load necessary modules
module load anaconda

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Change to the correct directory
cd /home/jztk7/Desktop/Drone

# Set environment variables for multi-GPU optimization
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Print GPU information
echo "GPU information:"
nvidia-smi

# Check PyTorch multi-GPU setup
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
print(f'Available GPUs: {list(range(torch.cuda.device_count()))}')
"

# Run with original high-resource settings for comparison
echo "Starting MULTI-GPU ablation study with original high resource allocation..."
echo "Resource allocation: 2x V100 GPUs, 150GB RAM, 7h time limit"
echo "This will be compared against the single GPU version for efficiency analysis"

# Run all configurations for comprehensive comparison
python ablation_study2.py config/config_5.json config/config_8.json config/config_10.json

echo "Multi-GPU ablation study completed at: $(date)"

# Show final GPU usage statistics
echo "Final GPU usage statistics:"
nvidia-smi

# Show memory usage
echo "Memory usage summary:"
free -h

# List the generated results
echo "Generated results:"
ls -la ablation_results/

# Show saved models
echo "Saved models in latest results directory:"
latest_dir=$(ls -t ablation_results/ | head -1)
if [ -n "$latest_dir" ]; then
    echo "Latest results directory: $latest_dir"
    ls -la "ablation_results/$latest_dir/"*.pth 2>/dev/null || echo "No model files found"
    echo "Total files in results directory:"
    ls -la "ablation_results/$latest_dir/" | wc -l
fi