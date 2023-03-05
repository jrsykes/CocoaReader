source activate convnext


python '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/AdaBoost.py' \
        --model_name 'CocoaNet18_AdaBoost' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'split_cocoa_images2_test' \
        --input_size 277 \
        --min_epochs 10 \
        --max_epochs 500 \
        --batch_size 32 \
        --patience 30 \
        --project_name 'CocoaNet'