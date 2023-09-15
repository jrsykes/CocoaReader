#!/bin/bash

#SBATCH --job-name=DisNet-RGB
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12  # Request 6 CPU cores


# Activate the torch4 environment
source activate torch3

# Set wandb directory
# export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"
export WANDB_DIR="/local/scratch/jrs596/WANDB_cache/"

# Set the root directory
# ROOT="/home/userfs/j/jrs596"
ROOT="/local/scratch/jrs596/"

python scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet-RGB' \
        --project_name 'DisNet-Pico-IR' \
        --root '/local/scratch/jrs596' \
        --data_dir 'dat/IR_RGB_Comp_data/RGB_split_1k' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --sweep \
        --arch 'DisNet' \
        --sweep_config '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/DisNet_pico-IR_config.yml' \
        --GPU 1 \
        --sweep_id 'pcyyxm20'
