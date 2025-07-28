#!/bin/bash
#SBATCH --job-name=drone_comparison
#SBATCH --output=logs/drone_comparison_%j.out
#SBATCH --error=logs/drone_comparison_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:V100-SXM2-32GB:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "==========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "==========================================="

# Load any required modules (adjust as needed for your system)
# module load python/3.9
module load cuda/11.8

# Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print Python and package versions
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "Available devices:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 'CPU only')"

echo ""
echo "Starting Unified Algorithm Comparison..."
echo "Expected to run 180 experiments (12 algorithms × 5 environments × 3 seeds)"
echo ""

# Run the full comparison (not quick version)
python run_full_comparison.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "==========================================="
    echo "Job completed successfully!"
    echo "End Time: $(date)"
    echo "==========================================="
    
    # Find the latest results directory
    LATEST_RESULTS=$(ls -td unified_comparison_results_* 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULTS" ]; then
        echo "Results saved in: $LATEST_RESULTS"
        echo ""
        echo "Summary Report:"
        cat "$LATEST_RESULTS/summary_report.md"
        echo ""
        echo "Statistical Analysis:"
        head -10 "$LATEST_RESULTS/statistical_analysis.csv"
    fi
else
    echo "==========================================="
    echo "Job failed with exit code: $?"
    echo "End Time: $(date)"
    echo "==========================================="
    exit 1
fi