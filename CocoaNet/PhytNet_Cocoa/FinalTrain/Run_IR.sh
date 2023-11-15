#!/bin/bash

#SBATCH --job-name=PhytNet-Cocoa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=70G
#SBATCH --cpus-per-task=12  # Request 6 CPU cores


# Activate the conda environment
source activate torch5

# Set wandb directory
export WANDB_DIR="/local/scratch/jrs596/WANDB_cache/"
# export WANDB_DIR="/scratch/staff/jrs596/WANDB_cache"


python scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhytNet-Cocoa' \
        --project_name 'PhytNet-Cocoa' \
        --root '/local/scratch/jrs596/' \
        --data_dir 'dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500' \
        --arch 'convnext_tiny' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 10 \
        --patience 20 \
        --GPU 1 \
        --custom_pretrained_weights 'dat/models/PhyloNet_V0_epoch_20_weights.pth' \
        --sweep_config '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/SweepConfig.yml' \
        --sweep_count 1000 \
        --sweep_id 'vkq7eq21' \


