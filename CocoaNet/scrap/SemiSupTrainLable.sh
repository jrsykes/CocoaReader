source activate convnext


python '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/SemiSupRun.py' \
        --model_name 'CocoaConvNext_SS' \
        --project_name 'CocoaConvNext_SS' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'EcuadorWebImages_EasyDif_FinalClean_SplitCompress/Easy' \
        --input_size 330 \
        --min_epochs 10 \
        --max_epochs 100 \
        --batch_size 32 \
        --patience 10 \
        --arch 'convnext_tiny' \
        --learning_rate 7.97193898713692e-05 \
        --weight_decay 0.00024296468323252175


