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
import numpy as np

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
parser.add_argument('--arch', type=str, default='resnet18',
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
    data_dir, _, initial_bias, _ = toolbox.setup(args)
    device = torch.device("cuda:2")
    criterion = nn.CrossEntropyLoss()

    # Initialize lists to store results
    train_metrics_dict = {'loss': [], 'f1': [], 'acc': [], 'precision': [], 'recall': [], 'BPR_F1': [], 'FPR_F1': [], 'Healthy_F1': [], 'WBD_F1': []}
    val_metrics_dict = {'loss': [], 'f1': [], 'acc': [], 'precision': [], 'recall': [], 'BPR_F1': [], 'FPR_F1': [], 'Healthy_F1': [], 'WBD_F1': []}
        
    for fold in range(10):
        print(f'Fold {fold}')
        
        model = toolbox.build_model(num_classes=4, arch=args.arch, config=None).to(device)
        

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                                weight_decay=args.weight_decay, eps=args.eps)

        
        wandb.init(project=args.project_name)
   
        # Create training and validation datasets using the current fold
        image_datasets = toolbox.build_datasets(input_size=args.input_size, data_dir=os.path.join(data_dir, f'fold_{fold}'))
    
        # Create dataloaders for the training and validation datasets
        dataloaders_dict = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=False),
            'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=False)
        }

        # Train the model and store the results
        _, _, _, _, run_name, best_train_metrics, best_val_metrics = train_model(
            model=model,
            args=args,
            optimizer=optimizer,
            device=device,
            dataloaders_dict=dataloaders_dict,
            criterion=criterion,
            patience=args.patience,
            initial_bias=initial_bias,
            input_size=None,
            n_tokens=None,
            batch_size=args.batch_size,
            AttNet=None,
            ANoptimizer=None
        )

        wandb.finish()

        # Store the results for this fold
        for metric in train_metrics_dict:
            train_metrics_dict[metric].append(best_train_metrics[metric])
        for metric in val_metrics_dict:
            val_metrics_dict[metric].append(best_val_metrics[metric])                                                   
                                                       
    #Divide all values in the dictionaries by 10 to get the mean
    mean_train_metrics_dict = {}
    mean_val_metrics_dict = {}

    for metric in train_metrics_dict:
        mean_train_metrics_dict[metric] = np.mean(train_metrics_dict[metric])
    for metric in val_metrics_dict:
        mean_val_metrics_dict[metric] = np.mean(val_metrics_dict[metric])

    #Calculate standard error metrics dict
    train_se_metrics_dict = {}
    val_se_metrics_dict = {}

    for metric in train_metrics_dict:
        train_se_metrics_dict[metric] = np.std(train_metrics_dict[metric]) / np.sqrt(len(train_metrics_dict[metric]))
    for metric in val_metrics_dict:
        val_se_metrics_dict[metric] = np.std(val_metrics_dict[metric]) / np.sqrt(len(val_metrics_dict[metric]))


    print()
    print(f'Mean train metrics: {mean_train_metrics_dict}')
    print(f'Standard error train metrics: {train_se_metrics_dict}')
    print()
    print(f'Mean val metrics: {mean_val_metrics_dict}')
    print(f'Standard error val metrics: {val_se_metrics_dict}')
    print()
          
    run = wandb.init(project=args.project_name)
    artifact = wandb.Artifact(run_name + '_results', type='dataset')

    # Log the results as wandb artifacts
    mean_dict = {'train_mean_metrics': mean_train_metrics_dict, 'val_mean_metrics': mean_val_metrics_dict,
                     'train_se_metrics': train_se_metrics_dict, 'val_se_metrics': val_se_metrics_dict}
    
    with open(run_name + '_results_dict.json', 'w') as f:
        json.dump(mean_dict, f)

    artifact.add_file(run_name + '_results_dict.json')
    run.log_artifact(artifact)

    wandb.finish()
    os.remove(run_name + '_results_dict.json')

train()