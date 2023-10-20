#!/bin/bash

#SBATCH --job-name=IR-RGB_cross-val
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --partition=gpu_big
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6  # Request 6 CPU cores


# Activate the conda environment
source activate torch5

# Set wandb directory
export WANDB_DIR="/local/scratch/jrs596/WANDB_cache/"


# Run the python script with srun
python '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/Torch_Custom_CNNs2.2.1_cross-val.py' \
        --project_name 'IR-RGB_cross-val' \
        --run_name 'ConvNext_tiny_CrossVal_IR' \
        --root '/local/scratch/jrs596' \
        --data_dir 'dat/IR_RGB_Comp_data/cross-val_IR' \
        --input_size 212 \
        --min_epochs 10 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --arch 'convnext_tiny' \
        --GPU 0
