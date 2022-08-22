#!/bin/bash

#SBATCH --partition=big

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=CocoaNet_INPretrained

# set number of GPUs
#SBATCH --gres=gpu:4

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
        --model_name 'CocoaNet_750kdim_ConvNeXt_INPreTrained' \
        --root '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat' \
        --data_dir 'split_cocoa_images' \
        --input_size 750 \
        --min_epochs 10 \
	--arch convnext_tiny \
        --batch_size 37 \
        --patience 10
        



