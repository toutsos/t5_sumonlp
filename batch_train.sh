#!/bin/bash
#SBATCH --job-name=custom_t5_training
#SBATCH --output=/home/angelos.toutsios.gr/data/Thesis_dev/nmt_model_train_custom/logs/log_%j.out  # Output log file
#SBATCH --error=/home/angelos.toutsios.gr/data/Thesis_dev/nmt_model_train_custom/logs/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32
#SBATCH --time=60:00:00             # Time limit (hh:mm:ss)
#SBATCH --partition=genai            # Specify the partition
#SBATCH --gres=gpu:1                 # Request 1 GPU

# . /etc/profile
# module load lib/cuda/9.0.176
# module load util/cuda-toolkit/12.0

# conda activate nmt_env  # Change to your desired conda environment

python -u train.py