
python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_ResDes18_1kdim.py' \
        --model_name 'CocoaNet18_quantised' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'split_cocoa_images' \
        --input_size 750 \
        --min_epochs 8 \
	--arch convnext_tiny \
        --batch_size 37 \
        --patience 6 \
        --custom_pretrained \
        --custom_pretrained_weights 'CocoaNet_750kdim_ConvNeXt_DNPreTrained_RecallWeighted.pkl' \
        --quantise 
        



