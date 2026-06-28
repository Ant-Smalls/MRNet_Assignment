#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_sgd
#SBATCH --output=job_outputs/eval_sgd_%j.out
#SBATCH --error=job_outputs/eval_sgd_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25274799@ucd.ie

# Change to the project directory
cd $SLURM_SUBMIT_DIR

# Load python module
module load python/3.10

# Activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt -q

echo "=========================================="
echo "Starting SGD Evaluation Job on HPC"
echo "Job ID:    $SLURM_JOB_ID"
echo "Start:     $(date)"
echo "GPU:       $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'No GPU found')"
echo "=========================================="

# Create outputs directory if missing
mkdir -p job_outputs/ results_sgd/

# Run evaluation on the SGD checkpoints
python code/src/evaluation.py \
    --condition all \
    --architectures baseline comparative \
    --data_modes uncropped cropped \
    --data_dir /scratch/25274799/mrnet_data \
    --checkpoint_dir checkpoints_sgd \
    --output_dir results_sgd/

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
