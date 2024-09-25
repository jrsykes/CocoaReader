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

python /users/jrs596/scripts/CocoaReader/CocoaNet/Pre-training/DFLoss_pre-train/FinalTrain/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhytNet-DFLoss-FAIGB' \
        --project_name 'Pre-training' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/FAIGB/FAIGB_700_30-10-23_split' \
        --input_size 308 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 42 \
        --patience 10 \
        --arch 'PhytNetV0' \
        --GPU 0 \
        --use_wandb \
        --save 
