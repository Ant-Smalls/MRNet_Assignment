#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=crop_dataset
#SBATCH --output=job_outputs/crop_%j.out
#SBATCH --error=job_outputs/crop_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25274799@ucd.ie

cd $SLURM_SUBMIT_DIR

module load python/3.10

if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

pip install -r requirements.txt -q

echo "=========================================="
echo "Starting Dataset Cropping on GPU"
echo "Job ID:    $SLURM_JOB_ID"
echo "Start:     $(date)"
echo "=========================================="

python3 -u code/src/modules/R-CNN_cropping_model/crop_mrnet_dataset.py \
    --model_path code/src/models/joint_detector.pth \
    --input_dir /scratch/25274799/mrnet_data/mrnet \
    --output_dir /scratch/25274799/mrnet_data/mrnet_cropped

JOB_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Exit code: $JOB_EXIT_CODE"
echo "=========================================="
