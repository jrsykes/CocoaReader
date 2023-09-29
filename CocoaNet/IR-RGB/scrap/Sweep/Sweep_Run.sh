source activate convnext

python '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/Sweep/Torch_Custom_CNNs2_1.py' \
        --project_name 'ConvNext_SimpleGC' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'IR_RGB_Comp_data/IR_split_400' \
        --input_size 400 \
        --eps 1e-6 \
        --weight_decay 1e-4 \
        --min_epochs 10 \
        --max_epochs 100 \
        --batch_size 21 \
        --patience 10 \
        --arch 'convnext_simple' \
        --sweep \
        --sweep_id '4ih8dpdy' \
        --sweep_count 1000 \
        --sweep_config '/home/userfs/j/jrs596/scripts/CocoaReader/utils/ConvNextSimpleCG_config.yml'

