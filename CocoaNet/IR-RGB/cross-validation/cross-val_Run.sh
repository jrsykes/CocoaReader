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
source activate torch4

# Set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

# Set the root directory
ROOT="/home/userfs/j/jrs596"

# Run the python script with srun
srun python 'scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/Torch_Custom_CNNs2.2.1_cross-val.py' \
        --project_name 'IR-RGB_cross-val' \
        --run_name 'DisNet_picoIR_earthy-sweep-4502' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/IR_RGB_Comp_data/cross-val_IR' \
        --min_epochs 10 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --arch 'DisNet_picoIR' \
        --GPU 0
