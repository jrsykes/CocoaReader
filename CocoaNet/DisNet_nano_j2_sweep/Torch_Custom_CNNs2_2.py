from __future__ import print_function
from __future__ import division

import yaml
import pprint
import torch
import torch.nn as nn
import wandb
import argparse
import sys
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
import toolbox
from training_loop import train_model
import random
import os
import json
from random_word import RandomWords
import time
import numpy as np


#Set seeds for reproducability
toolbox.SetSeeds()

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
parser.add_argument('--custom_pretrained', action='store_true', default=False,
                        help='Train useing specified pre-trained weights?')
parser.add_argument('--custom_pretrained_weights', type=str,
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
parser.add_argument('--arch', type=str, default='ConvNeXt_simple',
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

def train():

    run = wandb.init(project=args.project_name)

    data_dir, num_classes, initial_bias, device = toolbox.setup(args)

    # #load config dictionary form json file
    # if args.sweep_config is not None:
    #     with open(args.sweep_config) as f:
    #         model_config = yaml.safe_load(f)
    model_config = {'num_classes': num_classes, 'input_size': args.input_size,
                'stochastic_depth_prob': np.random.uniform(0.0001, 0.001), 'layer_scale': 0.3,
                'dim_1': random.randint(14,60), 'dim_2': random.randint(14,60), 
                'dim_3': random.randint(14,60),
                'nodes_1': random.randint(64,130), 'nodes_2': random.randint(64,130),
                'kernel_1': random.randint(1,7), 'kernel_2': random.randint(1,7),
                'kernel_3': random.randint(1,7), 'kernel_4': random.randint(1,7),
                'kernel_5': random.randint(1,7), 'kernel_6': random.randint(1,7),
                'kernel_7': random.randint(1,7), 'kernel_8': random.randint(1,7),
      }

    model = toolbox.build_model(num_classes, args, config=model_config)

    model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                            weight_decay=args.weight_decay, eps=args.eps)

    image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=args.input_size) #If images are pre compressed, use input_size=None, else use input_size=args.input_size

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=6, worker_init_fn=toolbox.worker_init_fn, drop_last=False) for x in ['train', 'val']}
    
    criterion = {'val': nn.CrossEntropyLoss(), 'train': toolbox.DynamicFocalLoss(delta=args.delta, dataloader=dataloaders_dict['train'])}
    
    trained_model, best_f1, best_f1_loss, best_f1_AIC, best_train_f1 = train_model(args=args, model=model, optimizer=optimizer, device=device, dataloaders_dict=dataloaders_dict, criterion=criterion, patience=args.patience, initial_bias=initial_bias, input_size=None, n_tokens=args.n_tokens, batch_size=args.batch_size, AttNet=None, ANoptimizer=None)
    
    
    
    wandb.finish()
    return trained_model, best_f1, best_f1_loss, best_f1_AIC, best_train_f1, model_config

if args.sweep == True:
    with open(args.sweep_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        sweep_config = config['sweep_config']
        sweep_config['metric'] = config['metric']
        sweep_config['parameters'] = config['parameters']

        print('Sweep config:')
        pprint.pprint(sweep_config)
        print()
        if args.sweep_id is None:
            sweep_id = wandb.sweep(sweep_config, project=args.project_name, entity="frankslab")
        else:
            sweep_id = args.sweep_id
        print("Sweep ID: ", sweep_id)
        print()

    wandb.agent(sweep_id,
            project=args.project_name, 
            function=train,
            count=args.sweep_count)
else:
    if args.run_name is None:
        run_name = RandomWords().get_random_word() + '_' + str(time.time())[-2:]
        wandb.init(project=args.project_name, name=run_name, mode="offline")
    else:
        wandb.init(project=args.project_name, name=args.run_name, mode="offline")
        run_name = args.run_name

    trained_model, best_f1, best_f1_loss, best_f1_AIC, best_train_f1, config = train()

    n_parameters = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)   
    
    config['n_parameters'] = n_parameters
    config['loss'] = best_f1_loss

    config['f1'] = best_f1
    config['AIC'] = best_f1_AIC
    config['train_f1'] = best_train_f1

    #make directory if it doesn't exist
    os.makedirs(os.path.join("/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/models/HypSweep/", args.project_name), exist_ok=True)
    out_file = os.path.join("/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/models/HypSweep/", args.project_name, run_name + ".json")
    
    
    # Save the updated dictionary back to the JSON file
    with open(out_file, 'w') as f:
        json.dump(config, f)