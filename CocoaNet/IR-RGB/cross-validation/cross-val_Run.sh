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

# Set wandb directory
# export WANDB_DIR="/local/scratch/jrs596/WANDB_cache/"


# Run the python script with srun
python scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/Torch_Custom_CNNs2.2.1_cross-val.py \
        --project_name 'PhytNet-CocoaIR-ablation' \
        --run_name 'PhytNet_CrossVal_IR' \
        --root '/home/userfs/j/jrs596' \
        --data_dir '/local/scratch/jrs596/dat/IR_RGB_Comp_data/cross-val_IR' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --arch 'PhytNetV0_ablation' \
        --GPU 0