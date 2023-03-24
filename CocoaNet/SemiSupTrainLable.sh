source activate convnext


python '/home/userfs/j/jrs596/scripts/CocoaReader/utils/Torch_Custom_CNNs2.py' \
        --model_name 'CocoaConvNext_sweep' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'EcuadorImage_LowRes_17_03_23_SureUnsure/Sure' \
        --min_epochs 10 \
        --max_epochs 500 \
        --batch_size 4 \
        --patience 5 \
        --project_name 'CocoaNet_token' \
        --arch 'convnext_tiny' \
        --sweep \
        --sweep_config '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/token_sweep_config.yml'

