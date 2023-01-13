source activate convnext


python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs_wandbSweep.py' \
        --model_name 'DisNet18_sweep' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'FAIGB_FinalSplit' \
        --min_epochs 10 \
        --max_epochs 50000 \
        --initial_batch_size 32 \
        --patience 50 \
        --min_batch_size 4 \
        --subset_classes_balance \
        --sweep \
        --sweep_id 'DisNet/fvxvfwqe' \
        --sweep_count 100