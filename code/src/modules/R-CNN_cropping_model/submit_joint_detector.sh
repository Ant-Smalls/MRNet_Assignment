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




# Create the cropped dataset 

#!/bin/bash -l
#SBATCH --job-name=crop_mrnet
# Request 1 node
#SBATCH -N 1
# GPU partition
#SBATCH --partition=gpu
# Request 1 GPU 
#SBATCH --gres=gpu:1
# Number of CPU cores
#SBATCH --ntasks-per-node=4
# Expected runtime: 2 hours 
#SBATCH -t 02:00:00
# Email notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=student@ucdconnect.ie
# Output logs
#SBATCH --output=job_output/crop_%j.log
#SBATCH --error=job_output/crop_%j.err

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory
cd /home/people/{studentnumber}/MRNet-assignment

echo "Current working directory: $(pwd)"
echo ""

# Activate virtual environment
source venv/bin/activate

mkdir -p job_output

# Print Python and CUDA info
echo "Python version: $(python3 --version)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "Number of GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU Name: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

echo "=========================================="
echo "Starting Crop Job at $(date)"
echo "=========================================="
echo ""

# Run script
python3 -u code/src/modules/R-CNN_cropping_model/crop_mrnet_dataset.py \
    --model_path code/src/modules/R-CNN_cropping_model/models/joint_detector.pth \
    --input_dir code/src/data/mrnet/ \
    --output_dir code/src/data/mrnet_cropped

JOB_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Exit code: $JOB_EXIT_CODE"
echo "=========================================="
echo ""

# Copy outputs to a timestamped folder for record-keeping
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
