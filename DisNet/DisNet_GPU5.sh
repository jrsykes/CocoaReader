source activate convnext


python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs_wandbSweep.py' \
        --model_name 'DisNet18_sweep' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'FAIGB_FinalSplit' \
        --min_epochs 20 \
        --initial_batch_size 32 \
        --patience 20 \
        --subset_classes_balance \
        --sweep \
        --sweep_id 'DisNet/jw8zyr8u' \
        --sweep_count 100