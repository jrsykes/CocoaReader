#!/bin/bash

#SBATCH --partition=big

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=test

# set number of GPUs
#SBATCH --gres=gpu:4

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jrs596@york.ac.uk

# run the application

module load python/anaconda3
module load cuda/11.2
#module load pytorch/1.9.0
#module load pytorch/1.12.1

#conda init bash
source ~/.bashrc
#conda activate convnext
source activate convnext2

export CODE_DIR='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts'      #PATH_TO_CODE_DIR
cd $CODE_DIR

wandb offline
wandb agent --count 1 SWEEP_ID

python 'CocoaReader/DisNet/Torch_Custom_CNNs.py' \
        --model_name 'DisNet18_v0.3' \
        --root '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat' \
        --data_dir 'test' \
        --input_size 224 \
        --min_epochs 1 \
        --arch 'resnet18' \
        --initial_batch_size 37 \
        --patience 2


