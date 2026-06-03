#!/bin/bash

# Sample Slurm job script for Galvani

#SBATCH -J CoT-training                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=0-23:59             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=prints0.4.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=errors0.4.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.straub@student.uni-tuebingen.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls

export IDENTIFIER_STR="CoT_encoder_mp0.4_4eps"
export MASK_PERCENTAGE=0.4

# Set Conda environment variables
export XDG_CACHE_HOME='/mnt/lustre/work/eickhoff/esx833/.conda/py-311-pytorch/cache'
export CONDA_PKGS_DIRS='/mnt/lustre/work/eickhoff/esx400/.conda/py-311-pytorch/cache'

source ~/.bashrc
conda activate $WORK/.conda/py-311-pytorch

# Compute Phase
srun python3 CoT_training_script.py  # srun will automatically pickup the configuration defined via `#SBATCH` and `sbatch` command line arguments

conda deactivate
