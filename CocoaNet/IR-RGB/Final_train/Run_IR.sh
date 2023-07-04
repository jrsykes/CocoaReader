
#set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

ROOT="/users/jrs596"

python scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet_pico_IR' \
        --project_name 'DisNet-Pico-IR' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_400' \
        --input_size 400 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 15 \
        --arch 'DisNet-pico' \
        --save 'both'



# export WANDB_DIR="/local/scratch/jrs596/dat/WANDB_DIR"

# ROOT="/home/userfs/j/jrs596"
# cd $ROOT

# python scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/Torch_Custom_CNNs2.2.1.py \
#         --model_name 'DisNet_pico_IR_DN' \
#         --project_name 'DisNet-Pico-IR' \
#         --root '/local/scratch/jrs596' \
#         --data_dir 'dat/IR_RGB_Comp_data/IR_split_500' \
#         --input_size 494 \
#         --min_epochs 15 \
#         --max_epochs 200 \
#         --batch_size 21 \
#         --patience 15 \
#         --arch 'DisNet-pico' \
#         --weights '/local/scratch/jrs596/dat/models/DisNet-pico-disnet_weights.pth' \
#         --save 'model'


