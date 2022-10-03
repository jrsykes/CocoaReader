
python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs.py' \
        --model_name 'CocoaNet18_quantised_test_fuse' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'split_cocoa_images' \
        --input_size 750 \
        --min_epochs 5 \
        --arch 'resnet18' \
        --batch_size 17 \
        --patience 3 \
        --custom_pretrained \
        --custom_pretrained_weights 'CocoaNet18_DN.pkl' \
        --quantise
