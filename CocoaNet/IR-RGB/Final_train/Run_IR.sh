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
source activate torch3

# Set wandb directory
export WANDB_DIR="/local/scratch/jrs596/WANDB_cache"

# Set the root directory
ROOT="/home/userfs/j/jrs596"
cd $ROOT

python scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet-IR-earthy-sweep-4502' \
        --project_name 'DisNet-Pico-IR' \
        --root '/local/scratch/jrs596' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_1k' \
        --arch 'DisNet' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --GPU 0 \
        --sweep \
        --sweep_config '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/SweepConfig.yml' \
        --sweep_count 1000 \
        --sweep_id 'm16w0jf6'


