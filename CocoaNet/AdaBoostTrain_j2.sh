#!/bin/bash

#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=CocoaConvNext

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jrs596@york.ac.uk

# run the application

module load python/anaconda3
module load cuda/11.2
module load pytorch/1.9.0

conda init bash
source ~/.bashrc
source activate convnext

export CODE_DIR='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts'      #PATH_TO_CODE_DIR
cd $CODE_DIR

python 'CocoaReader/CocoaNet/AdaBoost.py' \
        --model_name 'CocoaNet18_AdaBoost' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'split_cocoa_images2_test' \
        --input_size 277 \
        --min_epochs 10 \
        --max_epochs 500 \
        --batch_size 32 \
        --patience 30 \
        --project_name 'CocoaNet' \
        --arch 'resnet50'