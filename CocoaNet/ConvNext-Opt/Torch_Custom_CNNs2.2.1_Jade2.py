from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import argparse
import sys
import yaml
import os
import json
import wandb
import pprint
from torchvision.models.convnext import ConvNeXt, CNBlockConfig
import random

parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='test',
                        help='save name for model')
parser.add_argument('--project_name', type=str, default='test',
                        help='Name for wandb project')
parser.add_argument('--run_name', type=str, default=None,
                        help='Name for wandb run')
parser.add_argument('--sweep', action='store_true', default=False,
                        help='Run Waits and Biases optimisation sweep')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--WANDB_MODE', type=str, default='online',
                        help='WANDB_MODE for running offline')
parser.add_argument('--sweep_config', type=str, default=None,
                        help='.yml sweep configuration file')
parser.add_argument('--model_config', type=str, default=None,
                        help='.yml model configuration file')
parser.add_argument('--sweep_count', type=int, default=100,
                        help='Number of models to train in sweep')
parser.add_argument('--root', type=str, default='/local/scratch/jrs596/dat/',
                        help='location of all data')
parser.add_argument('--data_dir', type=str, default='test',
                        help='location of all data')
parser.add_argument('--save', action='store_true', default=False,
                        help='Do you want to save the model?')
parser.add_argument('--weights', type=str, default=None,
                        help='location of pre-trained weights')
parser.add_argument('--quantise', action='store_true', default=False,
                        help='Train with Quantization Aware Training?')
parser.add_argument('--batch_size', type=int, default=21,
                        help='Initial batch size')
parser.add_argument('--max_epochs', type=int, default=2,
                        help='n epochs before early stopping')
parser.add_argument('--min_epochs', type=int, default=10,
                        help='n epochs before loss is assesed for early stopping')
parser.add_argument('--patience', type=int, default=1,
                        help='n epochs to run without improvment in loss')
parser.add_argument('--beta', type=float, default=1.00,
                        help='minimum required per cent improvment in validation loss')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--l1_lambda', type=float, default=1e-5,
                        help='l1_lambda for regularization, Default:1e-5')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--eps', type=float, default=1e-6,
                        help='eps, Default:1e-8')
parser.add_argument('--batchnorm_momentum', type=float, default=1e-1,
                        help='Batch norm momentum hyperparameter for resnets, Default:1e-1')
parser.add_argument('--input_size', type=int, default=None,
                        help='image input size')
parser.add_argument('--delta', type=float, default=1.4,
                        help='delta for dynamic focal loss')
parser.add_argument('--arch', type=str, default=None,
                        help='Model architecture. resnet18, resnet50, resnext50, resnext101 or convnext_tiny')
parser.add_argument('--cont_train', action='store_true', default=False,
                        help='Continue training from previous checkpoint?')
parser.add_argument('--remove_batch_norm', action='store_true', default=False,
                        help='Deactivate all batchnorm layers?')
parser.add_argument('--split_image', action='store_true', default=False,
                        help='Split image into smaller chunks?')
parser.add_argument('--n_tokens', type=int, default=4,
                        help='Sqrt of number of tokens to split image into')
parser.add_argument('--criterion', type=str, default='crossentropy',
                        help='Loss function to use. DFLOSS or crossentropy')


args = parser.parse_args()
print(args)

sys.path.append(os.path.join(os.getcwd(), 'scripts/CocoaReader/utils'))
import toolbox
from training_loop import train_model





def train():

    wandb.init(project=args.project_name)
    #Set seeds for reproducability
    toolbox.SetSeeds()
    
    data_dir, num_classes, initial_bias, _ = toolbox.setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_dict = {'one': random.randint(66,96), 
                   'two': random.randint(100, 192),
                    'three': random.randint(192, 384),
                    'four': random.randint(384, 768),
                    'k_1': random.randint(1, 4),
                    'k_2': random.randint(1, 4),
                    'k_3': random.randint(1, 4),
                    'k_4': random.randint(1, 4),
                    'stochastic_depth_prob': random.uniform(0.001, 0.1)
                                            }
    
    block_setting = [
        CNBlockConfig(config_dict['one'], config_dict['two'], config_dict['k_1']),
        CNBlockConfig(config_dict['two'], config_dict['three'], config_dict['k_2']),
        CNBlockConfig(config_dict['three'], config_dict['four'], config_dict['k_3']),
        CNBlockConfig(config_dict['four'], None, config_dict['k_4']),
    ]

    model = ConvNeXt(block_setting=block_setting, stochastic_depth_prob=config_dict['stochastic_depth_prob'], layer_scale=1e-6, num_classes=num_classes)

    if args.weights is not None:
     # Load the state dict of the checkpoint
        state_dict = torch.load(args.weights, map_location=device)

        # Remove the weights for the final layer from the state dict
        state_dict.pop('fc3.weight')
        state_dict.pop('fc3.bias')

        # Load the state dict into the model, ignoring the missing keys
        model.load_state_dict(state_dict, strict=False)

    model = torch.compile(model)
    
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                            weight_decay=args.weight_decay, eps=args.eps)

    image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=args.input_size) #If images are pre compressed, use input_size=None, else use input_size=args.input_size

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=6, worker_init_fn=toolbox.worker_init_fn, drop_last=False) for x in ['train', 'val']}
    
    criterion = nn.CrossEntropyLoss()

    trained_model, best_f1, best_f1_loss, best_train_f1, run_name = train_model(args=args, model=model, optimizer=optimizer, device=device, dataloaders_dict=dataloaders_dict, criterion=criterion, patience=args.patience, initial_bias=initial_bias, input_size=None, n_tokens=args.n_tokens, batch_size=args.batch_size, AttNet=None, ANoptimizer=None)

    GFLOPs, params = toolbox.count_flops(model, (1, 3, args.input_size, args.input_size), device)

    config_dict['best_f1'] = best_f1
    config_dict['best_f1_loss'] = best_f1_loss
    config_dict['best_train_f1'] = best_train_f1
    config_dict['run_name'] = run_name
    config_dict['GFLOPs'] = GFLOPs
    config_dict['params'] = params

    #make dir for config dict
    os.mkdir(os.path.join(args.root, 'sweep_dicts'))

    #save config dict as json
    with open(os.path.join(args.root, 'sweep_dicts', wandb.run.name + '.json'), 'w') as fp:
        json.dump(config_dict, fp)
    print('Saved config dict to json')
        
    
    return trained_model, best_f1, best_f1_loss, best_train_f1



for i in range(100):
    train()