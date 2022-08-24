
python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs.py' \
        --model_name 'CocoaNext_DNPretrained' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'split_cocoa_images' \
        --input_size 75 \
        --min_epochs 8 \
	--arch 'resnext101' \
        --batch_size 7 \
        --patience 6 \
        --custom_pretrained \
        --custom_pretrained_weights 'DesNet_ResNext101.pkl'
        



