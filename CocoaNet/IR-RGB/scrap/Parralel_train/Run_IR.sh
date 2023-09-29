
source activate torch3
#set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

ROOT="/users/jrs596"

python scripts/CocoaReader/CocoaNet/IR-RGB/Parralel_train/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisRes_DeepStack-IR' \
        --project_name 'DisNet-Pico-IR' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_1k' \
        --input_size 300 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 20 \
        --arch 'Unified' \
        --save 'both'


