#!/bin/bash

# #SBATCH --job-name=IR-RGB_cross-val
# #SBATCH --output=slurm_output_%j.txt
# #SBATCH --error=slurm_error_%j.txt
# #SBATCH --partition=gpu_big
# #SBATCH --gres=gpu:1
# #SBATCH --time=48:00:00
# #SBATCH --mem=100G
# #SBATCH --cpus-per-task=6  # Request 6 CPU cores


# # Activate the torch4 environment
source activate torch4

# Set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

# Set the root directory
ROOT="/home/userfs/j/jrs596"
cd $ROOT

# python scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/Torch_Custom_CNNs2.2.1.py \
#         --model_name 'DisResNet' \
#         --project_name 'DisNet-Pico-IR' \
#         --root '/users/jrs596/scratch' \
#         --data_dir 'dat/IR_RGB_Comp_data/IR_split_1k' \
#         --min_epochs 15 \
#         --max_epochs 200 \
#         --batch_size 21 \
#         --patience 20 \
#         --sweep \
#         --sweep_config '/users/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/DisNet_pico-IR_config.yml' \
#         --GPU '0' \
#         --sweep_id 'cggl3s90' \
 

# # # Fetch GPU information using nvidia-smi
gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

# # Loop through each line to find idle GPUs
while read -r line; do
  # Extract GPU ID and memory usage
  gpu_id=$(echo $line | awk -F',' '{print $1}')
  memory_usage=$(echo $line | awk -F',' '{print $2}')

  # Check if the GPU is idle (memory usage is <= 5)
  if [[ $memory_usage -le 5 ]]; then
    echo "GPU $gpu_id is idle. Running Python script on this GPU in a new tmux window."

    # Run the Python command with the idle GPU in a new tmux window
    tmux new-window -n "GPU-$gpu_id" "module purge; \
        module load lang/Miniconda3; \
        module load system/CUDA/11.8.0; \
        conda init bash; \
        source ~/.bashrc; \
        conda activate torch4; \
        cd /users/jrs596; \
        export WANDB_DIR='/users/jrs596/scratch/WANDB_cache'; \
        python scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisResNet' \
        --project_name 'DisNet-Pico-IR' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_1k' \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --sweep \
        --sweep_config '/users/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/Arch_Sweep/DisNet_pico-IR_config.yml' \
        --GPU '$gpu_id' \
        --sweep_id 'ohrsq16a' > /users/jrs596/logs/GPU-$gpu_id.log 2>&1"
  fi
done <<< "$gpu_info"


##########################################################################################


      

