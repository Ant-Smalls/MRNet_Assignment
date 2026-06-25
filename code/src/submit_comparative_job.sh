#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=comparative
#SBATCH --output=job_outputs/comparative_%j.out
#SBATCH --error=job_outputs/comparative_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25205761@ucdconnect.ie

cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "Condition: $CONDITION | Plane: $PLANE | Data: $DATA_MODE | Arch: $COMP_ARCH"
echo "=========================================="

# Load python module
module load python/3.10

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Install dependencies (including torchxrayvision)
pip install -r requirements.txt -q

# Confirm GPU
echo ""
nvidia-smi
echo ""
echo "Python:  $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU:     $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

echo "=========================================="
echo "Starting training at $(date)"
echo "=========================================="

mkdir -p checkpoints

python3 -u code/src/train.py \
    --condition        $CONDITION \
    --plane            $PLANE \
    --architecture     comparative \
    --comparative_arch $COMP_ARCH \
    --data_mode        $DATA_MODE \
    --training_mode    two-phase \
    --epochs_phase1    10 \
    --epochs_phase2    20 \
    --lr_phase1        1e-3 \
    --lr_phase2        1e-4 \
    --weight_decay     1e-4 \
    --patience         10 \
    --batch_size       1 \
    --data_dir         code/src/data/ \
    --checkpoint_dir   checkpoints/

JOB_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Exit code: $JOB_EXIT_CODE"
echo "=========================================="
