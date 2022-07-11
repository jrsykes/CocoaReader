#!/bin/bash

#SBATCH --partition=big

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=ResDes18

# set number of GPUs
#SBATCH --gres=gpu:8

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
conda activate convnext

export CODE_DIR='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts'      #PATH_TO_CODE_DIR
cd $CODE_DIR


python 'CocoaReader/DisNet/Torch_ResDes18_1kdim.py' \
        --model_name 'DisNet_1kdim_HighRes_ConvNext' \
        --root '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat' \
        --data_dir 'FAIGB_clean_filtered_split' \
        --input_size 1000 \
        --min_epochs 1





