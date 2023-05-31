#source activate convnext

# python '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/Torch_Custom_CNNs2_1_cross-val.py' \
#         --project_name 'IR-RGB_sweep' \
#         --run_name 'RGB_cross-val' \
#         --root '/local/scratch/jrs596/dat/' \
#         --data_dir 'IR_RGB_Comp_data/cross-val_RGB' \
#         --input_size 322 \
#         --eps 0.00000001 \
#         --learning_rate 0.0001 \
#         --weight_decay 0.01 \
#         --batchnorm_momentum 0.1 \
#         --min_epochs 10 \
#         --max_epochs 100 \
#         --batch_size 21 \
#         --patience 10 \
#         --arch 'resnet18'

python '/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/Torch_Custom_CNNs2_1_cross-val.py' \
        --project_name 'IR-RGB_sweep' \
        --run_name 'IR_cross-val' \
        --root '/local/scratch/jrs596/dat/' \
        --data_dir 'IR_RGB_Comp_data/cross-val_IR' \
        --input_size 287 \
        --eps 0.00001 \
        --learning_rate 0.0001 \
        --weight_decay 0.0001 \
        --batchnorm_momentum 0.9 \
        --min_epochs 10 \
        --max_epochs 100 \
        --batch_size 21 \
        --patience 10 \
        --arch 'CGresnet18'