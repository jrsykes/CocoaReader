#!/bin/bash

#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00:00

# set name of job
#SBATCH --job-name=ConvNextOpt

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --array=1-3

# set maximum number of tasks to run in parallel
#SBATCH --ntasks=3

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


python /jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts/CocoaReader/scrap/test.py


