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

##SBATCH --array=1-40

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

root = "/jmain02/home/J2AD016/jjw02/jjs00-jjw02/"
cd $root

#set wandb directory
export WANDB_DIR=$root

python scripts/CocoaReader/CocoaNet/DisNet_nano_j2_sweep/Torch_Custom_CNNs2_2.py \
        --model_name 'DisNet-Nano' \
        --project_name 'DisNet-Nano' \
        --root '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat' \
        --data_dir 'EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Easy/' \
        --input_size 400 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 32 \
        --patience 10 \
        --learning_rate 1e-3 \
        --weight_decay 1e-4 \
        --arch 'DisNet_Nano'

