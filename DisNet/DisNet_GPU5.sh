
python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs_wandbSweep.py' \
        --model_name 'DisNet18_test' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'test' \
        --input_size 224 \
        --min_epochs 1 \
        --arch 'resnet18' \
        --initial_batch_size 8 \
        --patience 1 \
        --sweep_id 'DisNet/ey8egw8q'