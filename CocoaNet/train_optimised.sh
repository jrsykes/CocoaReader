source activate convnext


python '/home/userfs/j/jrs596/scripts/CocoaReader/utils/Torch_Custom_CNNs.py' \
        --model_name 'CocoaNet18_V1.3' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'split_cocoa_images2' \
        --input_size 277 \
        --min_epochs 10 \
        --max_epochs 500 \
        --batch_size 32 \
        --patience 30 \
        --min_batch_size 4 \
        --project_name 'CocoaNet'