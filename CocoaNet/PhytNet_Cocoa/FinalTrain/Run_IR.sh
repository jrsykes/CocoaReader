#!/bin/bash

#SBATCH --account=biol-cocoa-2023
#SBATCH --job-name=PhytNet-cocoa
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12  


# Activate the conda environment
source activate torch5


python /users/jrs596/scripts/CocoaReader/CocoaNet/PhytNet_Cocoa/FinalTrain/Torch_Custom_CNNs2.2.1.py \
        --model_name 'ResNet18-Cocoa-IN-PT' \
        --project_name 'PhytNet-Cocoa' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500_2/Working_Dir' \
        --input_size 375 \
        --min_epochs 15 \
        --max_epochs 100 \
        --batch_size 42 \
        --patience 10 \
        --arch 'resnet18' \
        --GPU 0 \
        --use_wandb \
        --save

