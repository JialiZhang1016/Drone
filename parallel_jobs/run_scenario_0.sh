#!/bin/bash
#SBATCH --job-name=scenario_0_5_loc_stable
#SBATCH --output=runs/output/scenario_0-%j.out
#SBATCH --error=runs/output/scenario_0-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100-SXM2-32GB:1
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=jztk7@mst.edu

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"

# GPU information
echo "GPU information:"
nvidia-smi

# Show PyTorch and CUDA information
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None; [print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo "Starting scenario 0: 5_loc_stable..."
echo "Parameters: Locations=5, WeatherProb=0.8, ExtremeProb=0.05"

# Run the single scenario
python run_single_scenario.py 0

echo "Scenario 0 completed at: $(date)"
echo "Final GPU usage statistics:"
nvidia-smi

echo "Memory usage summary:"
free -h