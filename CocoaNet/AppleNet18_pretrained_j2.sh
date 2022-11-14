#!/bin/bash

#SBATCH --partition=big

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=AppleNet50

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

python 'CocoaReader/DisNet/Torch_Custom_CNNs.py' \
        --model_name 'AppleNet50_v0.2' \
        --root '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat' \
        --data_dir 'PlantPathologyKaggle/dat' \
        --input_size 1120 \
        --min_epochs 2 \
        --arch 'resnet50' \
        --initial_batch_size 64 \
        --patience 20 \
        --custom_pretrained \
        --custom_pretrained_weights 'DisNet50_v0.3.pkl'