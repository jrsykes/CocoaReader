#set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

ROOT="/home/userfs/j/jrs596"


python 'scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/Torch_Custom_CNNs2.2.1_cross-val.py' \
        --project_name 'IR-RGB_cross-val' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/IR_RGB_Comp_data/cross-val_IR' \
        --input_size 478 \
        --min_epochs 10 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 15 \
        --arch 'DisNet-pico'