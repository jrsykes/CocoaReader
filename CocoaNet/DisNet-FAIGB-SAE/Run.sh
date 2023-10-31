#!/bin/bash

#SBATCH --job-name=DisNet-IR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=70G
#SBATCH --cpus-per-task=12  # Request 6 CPU cores


# Activate the conda environment
# conda activate torch5

# Set wandb directory
# export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"
export WANDB_DIR="/local/scratch/jrs596/WANDB_cache"


python /home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/DisNet-FAIGB-SAE/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet-FAIGB-SAE' \
        --project_name 'DisNet-FAIGB-SAE' \
        --root '/local/scratch/jrs596' \
        --data_dir 'dat/test5' \
        --input_size 536 \
        --min_epochs 15 \
        --max_epochs 120 \
        --batch_size 2 \
        --patience 20 \
        --arch 'PhytNet_SRAutoencoder' \
        --GPU 0 \
        # --save 'both' \
