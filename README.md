# CocoaReader
Automated disease detection and tracking in cocoa trees with computer vision

There are a few useful tools in this repo but the main feature is the CocoaReader/utils/Torch_Custom_CNNs.py script.
Useing a config file such as CocoaReader/CocoaNet/CocoaNetSweep.sh (shown below) one can quickly and easily train neural network for image classification while also being able to: 
  1. Swap between architectures such as ResNet18, ResNet50 or ConvNext tiny
  2. Train with quantisation aware training
  3. Perform a "Weights and Biases" hyperparameter optimisation sweep
  4. Continously subsample and balance your dataset during training to aboid problems of imbalance and overfitting
  5. Load custom pretrained weights
  6. Train with decaying batchsize for very fine tuning
  7. Disable features such as the batchnorm layers of ResNet

<code> 
python 'CocoaReader/utils/Torch_Custom_CNNs.py' \
        --model_name 'CocoaNet18_quantised' \
        --root <location of data dit> \
        --data_dir <data dir> \
        --input_size 750 \
        --min_epochs 5 \
        --arch 'resnet18' \
        --batch_size 17 \
        --patience 3 \
        --custom_pretrained \
        --custom_pretrained_weights 'CocoaNet18_DN.pkl' \
        --quantise <code>
