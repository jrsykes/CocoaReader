python 'scripts/CocoaReader/DisNet/Torch_Custom_CNNs.py' \
        --model_name 'AppleNet18' \
        --root '/local/scratch/jrs596/dat' \
        --data_dir 'PlantPathologyKaggle/dat' \
        --input_size 224 \
        --min_epochs 10 \
        --arch 'resnet18' \
        --batch_size 37 \
        --patience 50 \
        --custom_pretrained \
        --custom_pretrained_weights 'DisNet18_1120_reflect.pkl'


