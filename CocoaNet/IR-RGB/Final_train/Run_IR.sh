#!/bin/bash

#SBATCH --job-name=IR-RGB_cross-val
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --partition=gpu_big
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6  # Request 6 CPU cores


# Activate the conda environment
source activate torch5

# Set wandb directory
export WANDB_DIR="/local/scratch/jrs596/WANDB_cache/"
# export WANDB_DIR="/scratch/staff/jrs596/WANDB_cache"


python scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/Torch_Custom_CNNs2.2.1.py \
        --model_name 'convnext_tiny_IR_sweep' \
        --project_name 'DisNet-Pico-IR' \
        --root '/local/scratch/jrs596/' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_1k' \
        --arch 'convnext_tiny' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 10 \
        --patience 20 \
        --GPU 1 \
        --sweep_config '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/SweepConfig.yml' \
        --sweep_count 1000 \
        --sweep_id 'vkq7eq21' \


