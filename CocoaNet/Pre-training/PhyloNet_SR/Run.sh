#!/bin/bash

#SBATCH --account=biol-cocoa-2023
#SBATCH --job-name=PhyloNet_V0_1
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12  


# Activate the conda environment
source /users/jrs596/miniconda3/etc/profile.d/conda.sh
conda activate torch5


python /users/jrs596/scripts/CocoaReader/CocoaNet/Pre-training/PhyloNet_SR/Torch_Custom_CNNs2.2.1.py \
        --model_name 'PhytNet_PhyloPT_Flowers_NoL1_CosignSim' \
        --project_name 'Pre-training' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/flowers102_split' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 60 \
        --patience 20 \
        --arch 'PhytNetV0' \
        --GPU 0 \
        --eps 1e-6 \
        --l1_lambda 1e-6 \
        --save
        # --sweep_config '/users/jrs596/scripts/CocoaReader/CocoaNet/Pre-training/PhyloNet_SR/SweepConfig.yml' \
        # --sweep_id 'bgstul93'
