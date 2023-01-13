# CocoaReader
Automated disease detection and tracking in cocoa trees with computer vision

There area a few useful tools in this repo but the main feature is the CocoaReader/utils/Torch_Custom_CNNs.py script.
Useing a config file such as CocoaReader/CocoaNet/CocoaNetSweep.sh (shown below) one can quickly and easily train neural network for image classification while:
  A. Swapping between architectures such as ResNet18, ResNet50 or ConvNext tiny
  B. Training with quantisation aware training
  C. Perform a "Weights and Biases" hyperparameter optimisation sweep
  D. Continously subsample and balance your dataset during training to aboid problems of imbalance and overfitting
  E. Load custom pretrained weights
  F. Train with decaying batchsize for very fine tuning
  G. Disable features such as the batchnorm layers of ResNet

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
