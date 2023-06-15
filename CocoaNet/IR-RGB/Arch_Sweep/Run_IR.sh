#!/bin/bash

#SBATCH --partition=gpu

# set max wallclock time
#SBATCH --time=7-00:00:00

# set name of job
#SBATCH --job-name=ConvNextOpt

#SBATCH --ntasks=10

# set number of GPUs
#SBATCH --gres=gpu:1


##SBATCH --account=biol-cocoa-2023

# run the application

module purge
module load lang/Miniconda3 # for conda, if using venv you wont need this
module load system/CUDA/11.8.0

conda init bash
source ~/.bashrc

source activate convnext

ROOT="/users/jrs596"
cd $ROOT

#set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

ROOT="/home/userfs/j/jrs596"

python scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet-Pico-IR' \
        --project_name 'DisNet-Pico-IR' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_400' \
        --input_size 400 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 15 \
        --arch 'DisNet_pico-IR' \
        --sweep \
        --sweep_id '1h2glqp5' \
        --sweep_count 1000 \
        --sweep_config 'scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/DisNet_pico-IR_config.yml'
      


# export WANDB_DIR="/local/scratch/jrs596/dat/WANDB_DIR"

# ROOT="/home/userfs/j/jrs596"
# cd $ROOT
# python scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/Torch_Custom_CNNs2.2.1.py \
#         --model_name 'DisNet-Pico-IR' \
#         --project_name 'DisNet-Pico-IR' \
#         --root '/local/scratch/jrs596' \
#         --data_dir 'dat/IR_RGB_Comp_data/IR_split_400' \
#         --input_size 400 \
#         --min_epochs 15 \
#         --max_epochs 200 \
#         --batch_size 21 \
#         --patience 15 \
#         --arch 'DisNet_pico-IR' \
#         --sweep \
#         --sweep_count 1000 \
#         --sweep_config 'scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/DisNet_pico-IR_config.yml'