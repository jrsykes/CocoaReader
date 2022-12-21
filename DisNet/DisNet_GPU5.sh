source activate convnext

python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs_wandbSweep.py' \
        --model_name 'DisNet18_sweep' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'FAIGB_FinalSplit' \
        --min_epochs 10 \
        --arch 'resnet18' \
        --initial_batch_size 32 \
        --patience 20 \
        --sweep \
        --sweep_id 'DisNet/3gdlrogn' \
        --sweep_count 100