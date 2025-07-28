#!/bin/bash
#SBATCH --job-name=run_ablation_study2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:V100-SXM2-32GB:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=jztk7@mst.edu
#SBATCH --output=runs/output/Mill2_gpu-%j.out
#SBATCH --error=runs/output/Mill2_gpu-%j.err

# Load necessary modules
module load anaconda

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Change to the correct directory
cd /home/jztk7/Desktop/Drone

# Set environment variables for GPU optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print GPU information
echo "GPU information:"
nvidia-smi

# Check PyTorch GPU setup
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Run with GPU optimization
echo "Starting GPU-optimized ablation study..."
python ablation_study2.py config/config_5.json

echo "GPU-optimized ablation study completed at: $(date)"

# Show final GPU usage
echo "Final GPU usage:"
nvidia-smi