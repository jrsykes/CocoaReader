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


# python /users/jrs596/scripts/CocoaReader/CocoaNet/PhytNet_Cocoa/FinalTrain/Torch_Custom_CNNs2.2.1.py \
#         --model_name 'ResNet18-Cocoa-SemiSupervised_NotCocoa_DFLoss2' \
#         --project_name 'PhytNet-Cocoa-Final' \
#         --root '/users/jrs596/scratch' \
#         --data_dir 'dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca' \
#         --min_epochs 15 \
#         --max_epochs 200 \
#         --batch_size 42 \
#         --patience 10 \
#         --arch 'resnet18' \
#         --GPU 0 \
#         --use_wandb \
#         --save 

python /users/jrs596/scripts/CocoaReader/CocoaNet/PhytNet_Cocoa/FinalTrain/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhytNet-Cocoa-FullSupervised_redo' \
        --project_name 'PhytNet-Cocoa-Final' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 42 \
        --patience 10 \
        --arch 'PhytNetV0' \
        --GPU 0 \
        --use_wandb \
        --save \
        # --custom_pretrained_weights '/users/jrs596/scratch/models/PhytNet_SR_FAIGB1.pth'

