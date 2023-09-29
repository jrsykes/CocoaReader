# #!/bin/bash

# #SBATCH --partition=small

# #SBATCH --ntasks=20 

# # set max wallclock time
# #SBATCH --time=6-00:00:00

# # set name of job
# #SBATCH --job-name=ConvNextOpt

# # set number of GPUs
# #SBATCH --gres=gpu:1

# #SBATCH --array=1-40

# # set maximum number of tasks to run in parallel
# #SBATCH --ntasks=40

# # mail alert at start, end and abortion of execution
# #SBATCH --mail-type=ALL

# # run the application

# module load python/anaconda3
# module load cuda/11.2
# module load pytorch/1.9.0

# conda init bash
# source ~/.bashrc
# source activate convnext

# export WANDB_MODE=offline

# ROOT="/jmain02/home/J2AD016/jjw02/jjs00-jjw02"
# cd $ROOT

# #set wandb directory
# export WANDB_DIR=$ROOT

# python scripts/CocoaReader/CocoaNet/ConvNext-Opt/Torch_Custom_CNNs2.2.1.py \
#         --project_name 'ConvNextOpt' \
#         --root '/jmain02/home/J2AD016/jjw02/jjs00-jjw02' \
#         --data_dir 'dat/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Easy/' \
#         --input_size 400 \
#         --min_epochs 15 \
#         --max_epochs 200 \
#         --batch_size 32 \
#         --patience 10 \
#         --WANDB_MODE 'offline'

#
#G-Viging

source activate torch3

set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

ROOT="/users/jrs596"

python scripts/CocoaReader/CocoaNet/ConvNext-Opt/Torch_Custom_CNNs2.2.1_Viking.py \
        --project_name 'ConvNextOpt_IR' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_400' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 32 \
        --patience 10 \
        --save 'model' \
        --sweep \
        --sweep_config 'scripts/CocoaReader/CocoaNet/ConvNext-Opt/ConvNextOpt_Sweep.yaml' \
        #--sweep_id '7j6csyou' 
#
#GPU5

# source activate torch3

# export WANDB_DIR="/local/scratch/jrs596/dat/WANDB_DIR"

# ROOT="/home/userfs/j/jrs596"
# cd $ROOT


# python scripts/CocoaReader/CocoaNet/ConvNext-Opt/Torch_Custom_CNNs2.2.1.py \
#         --project_name 'ConvNextOpt' \
#         --root '/local/scratch/jrs596' \
#         --data_dir 'dat/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Easy/' \
#         --input_size 400 \
#         --min_epochs 15 \
#         --max_epochs 200 \
#         --batch_size 32 \
#         --patience 10 \
#         --sweep \
#         --sweep_config 'scripts/CocoaReader/CocoaNet/ConvNext-Opt/ConvNextOpt_Sweep.yaml' 