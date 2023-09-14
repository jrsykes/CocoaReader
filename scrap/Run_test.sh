#!/bin/bash

#SBATCH --job-name=IR-RGB_cross-val
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --partition=gpu_big
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6  # Request 6 CPU cores


# Activate the torch4 environment
source activate torch4

# Set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

# Set the root directory
ROOT="/home/userfs/j/jrs596"

python main_finetune.py \
    --model convnextv2_atto \
    --eval false \
    --input_size 224 \
    --data_path /users/jrs596/scratch/dat/IR_RGB_Comp_data/IR_split_1k \


