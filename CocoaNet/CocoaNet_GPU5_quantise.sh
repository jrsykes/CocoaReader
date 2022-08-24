
python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs.py' \
        --model_name 'CocoaNet18_quantised' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'split_cocoa_images' \
        --input_size 750 \
        --min_epochs 2 \
	--arch 'resnet18' \
        --batch_size 37 \
        --patience 10 \
        --custom_pretrained \
        --custom_pretrained_weights 'CocoaNet18.pkl' \
        --quantise


