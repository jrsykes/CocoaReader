
python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs_qunat.py' \
        --model_name 'DesNet_ResNext50' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'test' \
        --input_size 75 \
        --min_epochs 1 \
	--arch resnext50 \
        --batch_size 10 \
        --patience 3


