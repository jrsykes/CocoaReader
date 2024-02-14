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



python /users/jrs596/scripts/CocoaReader/PhytNet_ablation/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhytNet-Cocoa-ablation' \
        --project_name 'PhytNet-CocoaIR-ablation' \
        --root '/users/jrs596/scratch' \
        --data_dir '/users/jrs596/scratch/dat/IR_split' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 10 \
        --patience 10 \
        --arch 'PhytNetV0_ablation' \
        --GPU 0 \
        --save 
        # --sweep_config /users/jrs596/scripts/CocoaReader/PhytNet_ablation/PhytNet-Cocoa-config.yml \
        # --sweep_id '6vunncpl'
  
