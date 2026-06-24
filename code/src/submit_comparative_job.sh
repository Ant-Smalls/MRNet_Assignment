#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=comp_${CONDITION}_${PLANE}_${DATA_MODE}_${COMP_ARCH}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25205761@ucdconnect.ie

PROJECT_DIR=$HOME/MRNet_Assignment

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "Condition: $CONDITION | Plane: $PLANE | Data: $DATA_MODE | Arch: $COMP_ARCH"
echo "=========================================="

cd $PROJECT_DIR

# Activate virtual environment
source venv/bin/activate

# Confirm GPU is available
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
