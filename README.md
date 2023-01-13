# CocoaReader
Automated disease detection and tracking in cocoa trees with computer vision

There area a few useful tools in this repo but the main feature is the CocoaReader/utils/Torch_Custom_CNNs.py script.
Useing a config file such as CocoaReader/CocoaNet/CocoaNetSweep.sh (shown below) one can quickly and easily train neural network for image classification while:
  1. Swapping between architectures such as ResNet18, ResNet50 or ConvNext tiny
  2. Training with quantisation aware training
  3. Perform a "Weights and Biases" hyperparameter optimisation sweep
  4. Continously subsample and balance your dataset during training to aboid problems of imbalance and overfitting
  5. Load custom pretrained weights
  6. Train with decaying batchsize for very fine tuning
  7. Disable features such as the batchnorm layers of ResNet

<code>
python 'CocoaReader/DisNet/Torch_Custom_CNNs.py' \
        --model_name 'CocoaNet18_quantised' \
        --root <location of data file> \
        --data_dir 'split_cocoa_images' \
        --input_size 750 \
        --min_epochs 5 \
<code>
