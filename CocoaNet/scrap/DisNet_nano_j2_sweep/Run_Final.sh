# #!/bin/bash

# #SBATCH --partition=small

# #SBATCH --ntasks=12 

# # set max wallclock time
# #SBATCH --time=6-00:00:00

# # set name of job
# #SBATCH --job-name=ConvNextOpt

# # set number of GPUs
# #SBATCH --gres=gpu:1

# ##SBATCH --array=1-32

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

# python scripts/CocoaReader/CocoaNet/DisNet_nano_j2_sweep/Torch_Custom_CNNs2.2.1.py \
#         --model_name 'DisNet-Pico_optimised' \
#         --project_name 'DisNet-Pico-ArchSweep' \
#         --root '/jmain02/home/J2AD016/jjw02/jjs00-jjw02' \
#         --data_dir 'dat/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Easy/' \
#         --input_size 400 \
#         --min_epochs 15 \
#         --max_epochs 200 \
#         --batch_size 32 \
#         --patience 10 \
#         --learning_rate 1e-3 \
#         --weight_decay 1e-4 \
#         --arch 'DisNet_pico' \
#         --WANDB_MODE 'offline'



# export WANDB_MODE=offline

ROOT="/home/userfs/j/jrs596"
cd $ROOT

#set wandb directory
export WANDB_DIR=$ROOT

python scripts/CocoaReader/CocoaNet/DisNet_nano_j2_sweep/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet-Pico_optimised' \
        --project_name 'DisNet-Pico-ArchSweep' \
        --root '/local/scratch/jrs596/' \
        --data_dir 'dat/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Easy' \
        --input_size 400 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 32 \
        --patience 15 \
        --learning_rate 1e-3 \
        --weight_decay 1e-4 \
        --arch 'DisNet_pico'
