#!/bin/bash

#SBATCH --job-name=DisNet-IR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=70G
#SBATCH --cpus-per-task=12  # Request 6 CPU cores


# Activate the conda environment
source activate torch5

# Set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"
# export WANDB_DIR="/local/scratch/jrs596/WANDB_cache/"

# # Set the root directory
# ROOT="/users/jrs596/scratch"
# # ROOT="/local/scratch/jrs596/"
# cd $ROOT

python scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet-IR' \
        --project_name 'DisNet-Pico-IR' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_1k' \
        --input_size 285 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --arch 'DisNet' \
        --sweep_config '/users/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/DisNet_pico-IR_config.yml' \
        --GPU 0 \
        --sweep_count 10000
