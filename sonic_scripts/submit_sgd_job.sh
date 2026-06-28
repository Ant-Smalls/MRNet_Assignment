#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=sgd_train
#SBATCH --output=job_outputs/sgd_%j.out
#SBATCH --error=job_outputs/sgd_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25274799@ucd.ie

cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "Condition: $CONDITION | Plane: $PLANE | Data: $DATA_MODE | Arch: $ARCH | Optimizer: SGD"
echo "=========================================="

module load python/3.10

if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

pip install -r requirements.txt -q

echo ""
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

mkdir -p checkpoints_sgd

python3 -u code/src/train.py \
    --condition        $CONDITION \
    --plane            $PLANE \
    --architecture     $ARCH \
    --data_mode        $DATA_MODE \
    --training_mode    two-phase \
    --epochs_phase1    10 \
    --epochs_phase2    20 \
    --lr_phase1        1e-2 \
    --lr_phase2        1e-4 \
    --weight_decay     1e-4 \
    --patience         10 \
    --optimizer        sgd \
    --batch_size       1 \
    --data_dir         /scratch/25274799/mrnet_data \
    --checkpoint_dir   checkpoints_sgd/

JOB_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Exit code: $JOB_EXIT_CODE"
echo "=========================================="
