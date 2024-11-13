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



python /home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/PhytNet_ablation/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhytNet-Cocoa-ablation' \
        --project_name 'PhytNet-CocoaIR-ablation' \
        --root '/home/userfs/j/jrs596' \
        --data_dir '/local/scratch/jrs596/dat/IR_RGB_Comp_data/IR_split_1k' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 10 \
        --arch 'PhytNetV0_ablation' \
        --GPU 0 \
<<<<<<< HEAD
        --sweep_config /users/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/PhytNet_ablation/PhytNet-Cocoa-config.yml \
        --sweep_id 'g8o0v8ub'
=======
        --save

>>>>>>> refs/remotes/origin/main
  
