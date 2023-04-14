#!/bin/bash

#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00:00

# set name of job
#SBATCH --job-name=ConvNextOpt

# set number of GPUs and shards
#SBATCH --gres=gpu:1

#SBATCH --array=1-40

# set maximum number of tasks to run in parallel
#SBATCH --ntasks=40

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# run the application

module load python/anaconda3
module load cuda/11.2
module load pytorch/1.9.0

conda init bash
source ~/.bashrc
source activate convnext

export WANDB_MODE=offline


python /jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts/CocoaReader/utils/Torch_Custom_CNNs_j2_ConvNextOpt.py \
        --model_name 'CocoaNext_Opt' \
        --project_name 'ConvNext_Opt' \
        --root '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat' \
        --data_dir 'Ecuador_data/Late_split' \
        --input_size 320 \
        --min_epochs 10 \
        --max_epochs 200 \
        --batch_size 32 \
        --patience 10 \
        --learning_rate 7.97193898713692e-05 \
        --weight_decay 0.00024296468323252175 \
        --alpha 1.4 \
        --gamma 1.6


