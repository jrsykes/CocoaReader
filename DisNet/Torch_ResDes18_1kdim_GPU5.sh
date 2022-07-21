
python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_ResDes18_1kdim.py' \
        --model_name 'DisNet_1kdim_Binary_ResNet18' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'test' \
        --input_size 100 \
        --min_epochs 1 \
	--arch convnext_tiny \
        --batch_size 10 \
        --patience 3 \
        #--remove_batch_norm




