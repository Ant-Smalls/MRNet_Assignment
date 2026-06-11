#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=mrnet_train
#SBATCH --output=mrnet_train_%j.out
#SBATCH --error=mrnet_train_%j.err
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

# Run the training scripts sequentially for the 5 remaining models!
# We point them to the scratch directory where we uploaded the massive dataset
python train.py --condition meniscus --plane coronal --data_dir /scratch/25274799/mrnet_data/mrnet
python train.py --condition meniscus --plane sagittal --data_dir /scratch/25274799/mrnet_data/mrnet
python train.py --condition abnormal --plane axial --data_dir /scratch/25274799/mrnet_data/mrnet
python train.py --condition abnormal --plane coronal --data_dir /scratch/25274799/mrnet_data/mrnet
python train.py --condition abnormal --plane sagittal --data_dir /scratch/25274799/mrnet_data/mrnet
