export WANDB_DIR="/local/scratch/jrs596/dat/WANDB_DIR"

ROOT="/home/userfs/j/jrs596"
cd $ROOT

python scripts/CocoaReader/CocoaNet/IR-RGB/Final_train/Torch_Custom_CNNs2.2.1.py \
        --model_name 'DisNet-Pico-IR' \
        --project_name 'DisNet-Pico-IR' \
        --root '/local/scratch/jrs596' \
        --data_dir 'dat/IR_RGB_Comp_data/IR_split_400' \
        --input_size 400 \
        --min_epochs 15 \
        --max_epochs 200 \
        --batch_size 21 \
        --patience 10 \
        --arch 'DisNet_pico-IR' \
        --save