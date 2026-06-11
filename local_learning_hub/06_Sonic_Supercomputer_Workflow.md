# Sonic Supercomputer Workflow

This document outlines the end-to-end process for migrating our Deep Learning pipeline from Google Colab to the UCD Sonic High-Performance Computing (HPC) cluster.

## 1. Connecting to Sonic
You must be connected to the UCD VPN to access the supercomputer from off-campus.
To log in from your Mac terminal:
```bash
ssh 25274799@login.ucd.ie
```

## 2. Transferring Code and Data
The supercomputer has two main storage areas:
- `/home/people/25274799/`: For your code and scripts (small quota).
- `/scratch/25274799/`: High-speed, temporary storage for massive datasets (large quota).

### Getting Code onto Sonic
Log into Sonic and clone your GitHub repository:
```bash
cd ~
git clone https://github.com/Dublindeveloper/mrnet-learning.git
```

### Getting Data onto Sonic
From your **local Mac terminal**, use `scp` or `rsync` to securely copy data to the Sonic scratch directory:
```bash
rsync -avz --progress /Users/jinishrajan/ucd/Medimaging/code/src/data/mrnet/ 25274799@login.ucd.ie:/scratch/25274799/mrnet_data/mrnet/
```

## 3. SLURM Supercomputer Commands
Sonic uses a job scheduler called SLURM. You cannot run heavy Python code directly on the login node. You must submit a "job script" that SLURM puts in a queue for a GPU node.

- **Submit a Job:** `sbatch submit_mrnet.sh` (Returns a Job ID like `421691`)
- **Check Queue Status:** `squeue -u 25274799` (Look for 'R' = Running, 'PD' = Pending)
- **Cancel a Job:** `scancel 421691`
- **Live Output (Console Log):** `tail -f mrnet_train_421691.out`
- **Error Log:** `cat mrnet_train_421691.err`

## 4. The SLURM Script (`submit_mrnet.sh`)
This script tells SLURM what resources we need and what code to run.

**Critical Requirements for PyTorch:**
1. You **must** request a GPU device using `#SBATCH --gres=gpu:1`. If you only ask for the partition (`--partition=gpu`), SLURM will hide the GPUs from your code and PyTorch will default to the CPU.
2. The virtual environment (`venv`) must be activated *inside* the script before running Python.
3. The training script must point to the `/scratch` directory where the data lives.

### Example Submission Script:
```bash
#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=mrnet_train
#SBATCH --output=mrnet_train_%j.out
#SBATCH --error=mrnet_train_%j.err

cd $SLURM_SUBMIT_DIR/code/src

# Load UCD's python module
module load python/3.10

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install -r ../../requirements.txt

# Run Python training code pointing to the scratch data
python train.py --condition meniscus --plane coronal --data_dir /scratch/25274799/mrnet_data/mrnet
```

## 5. Avoiding Common Pitfalls
- **Missing Modules:** A raw Linux server doesn't have packages pre-installed like Colab does. If you get a `ModuleNotFoundError` (like we did with `tqdm`), explicitly add it to your `requirements.txt`.
- **Saving Checkpoints:** The supercomputer does NOT natively sync with your personal Google Drive. Model checkpoints `.pth` will be saved internally in the Sonic folder (`~/mrnet-learning/code/src/local_learning_hub/checkpoints`). You must download them back to your Mac manually using `scp` when training is finished.
- **Resuming Jobs:** Sonic doesn't know what you finished on Colab. You must explicitly tell your `submit_mrnet.sh` script which exact models to run, rather than looping over all of them.
