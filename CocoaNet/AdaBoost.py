#%%

#import Torch_Custom_CNNs.py from ~/scripts/CocoaReader/utils
import sys
import argparse
import os
import pandas as pd
import torch
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/Eval')
from Torch_Custom_CNNs2 import train
from AdaBoostEval import eval

parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='model',
                        help='save name for model')
parser.add_argument('--project_name', type=str, default=None,
                        help='Name for wandb project')
parser.add_argument('--sweep', action='store_true', default=False,
                        help='Run Waits and Biases optimisation sweep')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--sweep_config', type=str, default=None,
                        help='.yml sweep configuration file')
parser.add_argument('--sweep_count', type=int, default=100,
                        help='Number of models to train in sweep')
parser.add_argument('--root', type=str, default='/local/scratch/jrs596/dat',
                        help='location of all data')
parser.add_argument('--data_dir', type=str, default='test',
                        help='location of all data')
parser.add_argument('--custom_pretrained', action='store_true', default=False,
                        help='Train useing specified pre-trained weights?')
parser.add_argument('--custom_pretrained_weights', type=str,
                        help='location of pre-trained weights')
parser.add_argument('--quantise', action='store_true', default=False,
                        help='Train with Quantization Aware Training?')
parser.add_argument('--batch_size', type=int, default=32,
                        help='Initial batch size')
parser.add_argument('--max_epochs', type=int, default=500,
                        help='n epochs before early stopping')
parser.add_argument('--min_epochs', type=int, default=10,
                        help='n epochs before loss is assesed for early stopping')
parser.add_argument('--patience', type=int, default=1,
                        help='n epochs to run without improvment in loss')
parser.add_argument('--beta', type=float, default=1.00,
                        help='minimum required per cent improvment in validation loss')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps, Default:1e-8')
parser.add_argument('--input_size', type=int, default=224,
                        help='image input size')
parser.add_argument('--arch', type=str, default='resnet18',
                        help='Model architecture. resnet18, resnet50, resnext50, resnext101 or convnext_tiny')
parser.add_argument('--cont_train', action='store_true', default=False,
                        help='Continue training from previous checkpoint?')
parser.add_argument('--remove_batch_norm', action='store_true', default=False,
                        help='Deactivate all batchnorm layers?')


args = parser.parse_args()

weights_dict = {}

for dir_ in os.listdir(os.path.join(args.root, args.data_dir, "train")):
    for filename in os.listdir(os.path.join(args.root, args.data_dir, "train", dir_)):
        weights_dict[filename] = 1



for i in range(5):
#train the model
    print()
    print("Major epoch: ", str(i))
    model, image_datasets = train(weights_dict=weights_dict, args_override = args)
    PATH = "/local/scratch/jrs596/dat/models/AdaBoost/" + args.model_name + str(i)
    torch.save(model.module, PATH + '.pth') 

    weights_dict, F1, auc, acc, loss = eval(model=model, image_datasets=image_datasets['train'], weights_dict=weights_dict)

# %%
