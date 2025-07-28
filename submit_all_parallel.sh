#!/bin/bash
# Submit all 10 scenarios to run in parallel on different GPU nodes

echo "==========================================================================="
echo "PARALLEL ABLATION STUDY SUBMISSION"
echo "Submitting 10 scenarios to run simultaneously on different GPU nodes"
echo "==========================================================================="

# Array to store job IDs
declare -a job_ids

# Submit all 10 scenarios
for i in {0..9}; do
    echo "Submitting scenario $i..."
    job_id=$(sbatch parallel_jobs/run_scenario_$i.sh | awk '{print $4}')
    job_ids[$i]=$job_id
    echo "  Scenario $i submitted with Job ID: $job_id"
    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo ""
echo "==========================================================================="
echo "ALL JOBS SUBMITTED SUCCESSFULLY"
echo "==========================================================================="
echo "Job IDs:"
for i in {0..9}; do
    echo "  Scenario $i: ${job_ids[$i]}"
done

echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER'"
echo ""
echo "Check specific job logs:"
for i in {0..9}; do
    echo "  Scenario $i: tail -f runs/output/scenario_${i}-${job_ids[$i]}.out"
done

echo ""
echo "Check GPU usage across nodes:"
echo "  sinfo -p gpu -o '%N %T %G %C'"
echo ""
echo "==========================================================================="