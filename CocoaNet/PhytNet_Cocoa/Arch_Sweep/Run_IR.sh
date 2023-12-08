#!/bin/bash

#SBATCH --account=biol-cocoa-2023
#SBATCH --job-name=PhytNet-cocoa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12  


# Activate the conda environment
source activate torch5


python /users/jrs596/scripts/CocoaReader/CocoaNet/PhytNet_Cocoa/Arch_Sweep/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhytNet-Cocoa' \
        --project_name 'PhytNet-Cocoa' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Working_Dir' \
        --min_epochs 15 \
        --max_epochs 100 \
        --batch_size 42 \
        --patience 10 \
        --arch 'PhytNetV0' \
        --GPU 0 \
        --sweep_config '/users/jrs596/scripts/CocoaReader/CocoaNet/PhytNet_Cocoa/Arch_Sweep/PhytNet-Cocoa-config.yml' \
        --sweep_count 10000 \
        --sweep_id 'ckkegjf5' \

