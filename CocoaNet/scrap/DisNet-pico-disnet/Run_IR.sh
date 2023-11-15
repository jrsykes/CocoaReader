source activate torch3

set wandb directory
export WANDB_DIR="/users/jrs596/scratch/WANDB_cache"

ROOT="/users/jrs596"

python scripts/CocoaReader/CocoaNet/DisNet-pico-disnet/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet-pico-disnet' \
        --project_name 'DisNet-Pico-disnet' \
        --root '/users/jrs596/scratch' \
        --data_dir 'dat/FAIGB_494' \
        --input_size 277 \
        --min_epochs 15 \
        --max_epochs 1 \
        --batch_size 32 \
        --patience 15 \
        --arch 'efficientnet_b0' \
        --save 'both'
