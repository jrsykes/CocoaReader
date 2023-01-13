#!/bin/bash

#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=72:00:00

# set name of job
#SBATCH --job-name=ResDes18

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
conda activate NVAE

export CODE_DIR='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts/CocoaReader/ResNet'      #PATH_TO_CODE_DIR
cd $CODE_DIR
python Torch_ResDes18_1kdim.py --model_name 'ResDes18_750kdim_HighRes_PNPFiltered_WeightedLoss' --root '/scratch/staff/jrs596/dat' \
        --data_dir 'Forestry_ArableImages_GoogleBing_Final' --input_size 750




