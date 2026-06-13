#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=joint_detector
#SBATCH --output=joint_detector_%j.out
#SBATCH --error=joint_detector_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25274799@ucd.ie

# Change to the working directory
cd $SLURM_SUBMIT_DIR/code/src

# Load python module
module load python/3.10

# Create and activate a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Install dependencies (only installs if not already installed)
pip install -r ../../requirements.txt

# Run the Faster R-CNN joint detector training script!
# The -u flag forces python to unbuffer output so you can see live logs in the .out file
python -u train_joint_detector.py \
    --csv ../../joint_annotations.csv \
    --img_dir ../../annotation_samples \
    --epochs 15
