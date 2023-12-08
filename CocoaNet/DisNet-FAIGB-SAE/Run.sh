#!/bin/bash

#SBATCH --account=biol-cocoa-2023
#SBATCH --job-name=PhyloNet_V0_1
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12  


# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch5


python ~/scripts/CocoaReader/CocoaNet/DisNet-FAIGB-SAE/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhyloNet_V0_test' \
        --project_name 'DisNet-FAIGB-SAE' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/FAIGB/FAIGB_700_30-10-23_split' \
        --input_size 64 \
        --min_epochs 15 \
        --max_epochs 1000 \
        --batch_size 6 \
        --patience 20 \
        --arch 'PhytNet_SRAutoencoder' \
        --GPU 0 \
