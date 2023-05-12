#%%

#import Torch_Custom_CNNs.py from ~/scripts/CocoaReader/utils
import sys
import argparse
import os
import pandas as pd
import torch
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/Eval')
from Torch_Custom_CNNs2_1 import train
from EvalLable import eval
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='test',
                        help='save name for model')
parser.add_argument('--project_name', type=str, default=None,
                        help='Name for wandb project')
parser.add_argument('--run_name', type=str, default=None,
                        help='Name for wandb run')
parser.add_argument('--sweep', action='store_true', default=False,
                        help='Run Waits and Biases optimisation sweep')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--sweep_config', type=str, default=None,
                        help='.yml sweep configuration file')
parser.add_argument('--sweep_count', type=int, default=100,
                        help='Number of models to train in sweep')
parser.add_argument('--root', type=str, default='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat',
                        help='location of all data')
parser.add_argument('--data_dir', type=str, default='test',
                        help='location of all data')
parser.add_argument('--custom_pretrained', action='store_true', default=False,
                        help='Train useing specified pre-trained weights?')
parser.add_argument('--custom_pretrained_weights', type=str,
                        help='location of pre-trained weights')
parser.add_argument('--quantise', action='store_true', default=False,
                        help='Train with Quantization Aware Training?')
parser.add_argument('--batch_size', type=int, default=4,
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
parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps, Default:1e-8')
parser.add_argument('--input_size', type=int, default=277,
                        help='image input size')
parser.add_argument('--alpha', type=float, default=1.4,
                        help='alpha for dynamic focal loss')
parser.add_argument('--gamma', type=float, default=1.6,
                        help='gamma for dynamic focal loss')
parser.add_argument('--arch', type=str, default='convnext_tiny',
                        help='Model architecture. resnet18, resnet50, resnext50, resnext101 or convnext_tiny')
parser.add_argument('--cont_train', action='store_true', default=False,
                        help='Continue training from previous checkpoint?')
parser.add_argument('--remove_batch_norm', action='store_true', default=False,
                        help='Deactivate all batchnorm layers?')
parser.add_argument('--split_image', action='store_true', default=False,
                        help='Split image into smaller chunks?')
parser.add_argument('--n_tokens', type=int, default=4,
                        help='Sqrt of number of tokens to split image into')

args = parser.parse_args()


major_epoch = 0
moved_count = 1
model = models.convnext_tiny(weights = ConvNeXt_Tiny_Weights, input_size = args.input_size)
in_feat = model.classifier[2].in_features
model.classifier[2] = torch.nn.Linear(in_feat, 4)
model = torch.nn.DataParallel(model).cuda()

root = os.path.dirname(os.path.join(args.root, args.data_dir))

while moved_count > 0:
#train the model
    print()
    major_epoch += 1
    print('Major epoch: ', str(major_epoch))
    trained_model, f1, loss = train(args_override = args, model = model)
    PATH = "/local/scratch/jrs596/dat/models/SemiSup/" + args.model_name + str(major_epoch)
    torch.save(trained_model.module, PATH + '.pth') 

    moved_count = eval(root=root, model=trained_model, moved_count=moved_count, input_size=args.input_size, Diff_Unsure = 'Difficult')
    if moved_count == 0:
        moved_count = eval(root=root, model=trained_model, moved_count=moved_count, input_size=args.input_size, Diff_Unsure = 'Unsure')

    



# %%
