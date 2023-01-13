source activate convnext


python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs.py' \
        --model_name 'CocoaNet18_sweep' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'split_cocoa_images' \
        --min_epochs 10 \
        --max_epochs 500 \
        --initial_batch_size 32 \
        --patience 20 \
        --min_batch_size 4 \
        --sweep \
        --sweep_config 'CocoaNetSweepConfig.yml' \
        --sweep_count 100 \
        --project_name 'CocoaNet'