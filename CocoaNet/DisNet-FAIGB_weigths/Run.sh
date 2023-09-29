#!/bin/bash

#SBATCH --job-name=DisNet-IR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=70G
#SBATCH --cpus-per-task=12  # Request 6 CPU cores


# Activate the conda environment
source activate torch5

# Set wandb directory
# export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"
export WANDB_DIR="/local/scratch/jrs596/WANDB_cache"


python /home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/DisNet-FAIGB_weigths/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet' \
        --project_name 'DisNet-FAIGB_weigths' \
        --root '/local/scratch/jrs596' \
        --data_dir 'dat/FAIGB/FAIGB_FinalSplit_700' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 42 \
        --patience 20 \
        --arch 'DisNetV1_2' \
        --GPU 0 \
        --sweep_count 10000 \
        --sweep_config '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/DisNet-FAIGB_weigths/DisNet_V1_3_config.yml' \
        --sweep_id 'xvc1dc99' \

