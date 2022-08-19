
python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_ResDes18_1kdim.py' \
        --model_name 'CocoaNet_750kdim_ConvNeXt_PreTrained' \
        --root '/scratch/staff/jrs596/dat' \
        --data_dir 'test' \
        --input_size 750 \
        --min_epochs 37 \
	--arch convnext_tiny \
        --batch_size 10 \
        --patience 6 \
        --pretrained \
        --pretrained_weights 'DisNet_1kdim_HighRes_ConvNext.pkl'
        



