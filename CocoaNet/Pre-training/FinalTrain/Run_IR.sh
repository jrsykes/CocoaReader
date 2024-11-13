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

python /users/jrs596/scripts/CocoaReader/CocoaNet/Pre-training/FinalTrain/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhytNet-Cocoa-SemiSupervised_DFLoss-PhyloPreTrained' \
        --project_name 'Pre-training' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca' \
        --input_size 308 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 42 \
        --patience 10 \
        --arch 'PhytNetV0' \
        --GPU 0 \
        --use_wandb \
        --save \
        --custom_pretrained_weights '/users/jrs596/scratch/models/PhytNet_PhyloPT_Flowers_NoL1_CosignSim.pth' 

