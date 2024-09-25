#!/bin/bash

#SBATCH --account=biol-cocoa-2023
#SBATCH --job-name=PhytNet_Cocoa_CrossVal
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12  


# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch5

# Run the python script with srun
python '/users/jrs596/scripts/CocoaReader/CocoaNet/PhytNet_Cocoa/cross-validation/Torch_Custom_CNNs2.2.1_cross-val.py' \
        --project_name 'PhytNet_Cocoa_CrossVal' \
        --run_name 'PhytNet_Cocoa_CrossVal' \
        --root '/local/scratch/jrs596' \
        --data_dir 'dat/IR_RGB_Comp_data/cross-val_IR' \
        --input_size 212 \
        --min_epochs 10 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --arch 'convnext_tiny' \
        --GPU 0
