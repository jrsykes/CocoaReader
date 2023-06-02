source activate convnext

python '/home/userfs/j/jrs596/scripts/CocoaReader/scrap/Torch_Custom_tmp.py' \
        --project_name 'test' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'IR_RGB_Comp_data/RGB_split_400' \
        --input_size 400 \
        --eps 1e-6 \
        --weight_decay 1e-4 \
        --min_epochs 10 \
        --max_epochs 100 \
        --batch_size 21 \
        --patience 10 \
        --arch 'resnet18' \
        --delta 0.1 \

