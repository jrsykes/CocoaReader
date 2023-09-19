from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import argparse
import sys
import yaml
import os
import wandb
import pprint


parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='test',
                        help='save name for model')
parser.add_argument('--project_name', type=str, default='test',
                        help='Name for wandb project')
parser.add_argument('--run_name', type=str, default=None,
                        help='Name for wandb run')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--WANDB_MODE', type=str, default='online',
                        help='WANDB_MODE for running offline')
parser.add_argument('--sweep_config', type=str, default=None,
                        help='.yml sweep configuration file')
parser.add_argument('--model_config', type=str, default=None,
                        help='.yml model configuration file')
parser.add_argument('--sweep_count', type=int, default=1000,
                        help='Number of models to train in sweep')
parser.add_argument('--root', type=str, default='/local/scratch/jrs596/dat/',
                        help='location of all data')
parser.add_argument('--data_dir', type=str, default='test',
                        help='location of all data')
parser.add_argument('--save', type=str, default=None,
                        help='save "model", "weights" or "both" ?')
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
parser.add_argument('--GPU', type=str, default='0',
                        help='Which GPU device to use')


args = parser.parse_args()
print(args)

sys.path.append(os.path.join(os.getcwd(), 'scripts/CocoaReader/utils'))
import toolbox
from training_loop import train_model


def train():

    run = wandb.init(project=args.project_name, settings=wandb.Settings(_service_wait=300))
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    wandb.save(os.path.join(script_dir, '*'), base_path=script_dir) 

    #Set seeds for reproducability
    toolbox.SetSeeds(42)

    data_dir, num_classes, initial_bias, _ = toolbox.setup(args)
    device = torch.device("cuda:" + args.GPU)

    #define config dictionary with wandb
    config = {
        'num_classes': 8,
        'input_size': args.input_size,
        'drop_out': 0.2,
        'drop_out2': 0.2,
        'dim_1': wandb.config.dim_1, 
        'dim_2': wandb.config.dim_2, 
        'nodes_1': wandb.config.nodes_1, 
        'nodes_2': wandb.config.nodes_2,
        'kernel_1': wandb.config.kernel_1, 
        'kernel_2': wandb.config.kernel_2,
        'kernel_3': wandb.config.kernel_3, 
        'kernel_4': wandb.config.kernel_4,
        'kernel_5': wandb.config.kernel_5, 
        'kernel_6': wandb.config.kernel_6,
        'beta1': wandb.config.beta1,
        'beta2': wandb.config.beta2,
        'learning_rate': wandb.config.learning_rate
    }
        
    model = toolbox.build_model(num_classes=config['num_classes'], arch=args.arch, config=config).to(device)
  
    image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=config['input_size']) #If images are pre compressed, use input_size=None, else use input_size=args.input_size

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=6, worker_init_fn=toolbox.worker_init_fn, drop_last=False) for x in ['train', 'val']}
    
    criterion = nn.CrossEntropyLoss()
    
    input_size = torch.Size([3, wandb.config.input_size, wandb.config.input_size])
    inputs = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        model(inputs)
    
    GFLOPs, n_params = toolbox.count_flops(model=model, device=device, input_size=input_size)
    del model
    print()
    print('GFLOPs: ', GFLOPs, 'n_params: ', n_params)

    if GFLOPs < 6 and n_params < 50000000:
        # model = toolbox.build_model(num_classes=None, arch='Unified', config=config2).to(device)   
        model = toolbox.build_model(num_classes=config['num_classes'], arch=args.arch, config=config).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate,
                                        weight_decay=args.weight_decay, eps=args.eps)
        
        trained_model, best_f1, best_f1_loss, best_train_f1, run_name, _, _ = train_model(args=args, model=model, optimizer=optimizer, device=device, dataloaders_dict=dataloaders_dict, criterion=criterion, patience=args.patience, initial_bias=initial_bias, input_size=None, n_tokens=args.n_tokens, batch_size=args.batch_size, AttNet=None, ANoptimizer=None)
        config['Run_name'] = run_name
        
    else: 
        print()
        print('Model too large, aborting training')
        print()
        run.log({'Status': 'aborted'})  # Log the status as 'aborted'
        run.finish()  # Finish the run

        trained_model, best_f1, best_f1_loss, best_train_f1, config = None, None, None, None, None


    return trained_model, best_f1, best_f1_loss, best_train_f1, config

os.environ["WANDB__SERVICE_WAIT"] = "300"
if args.sweep_config != None:
    with open(args.sweep_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        sweep_config = config['sweep_config']
        sweep_config['metric'] = config['metric']
        sweep_config['parameters'] = config['parameters']
    
        print('Sweep config:')
        pprint.pprint(sweep_config)
        if args.sweep_id is None:
            sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name, entity="frankslab")
        else:
            sweep_id = args.sweep_id
        print("Sweep ID: ", sweep_id)
        print()
    
    wandb.agent(sweep_id,
            project=args.project_name, 
            function=train,
            count=args.sweep_count)
else:
    train()