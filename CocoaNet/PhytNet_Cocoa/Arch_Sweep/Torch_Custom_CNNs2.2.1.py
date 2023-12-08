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
import shutil
from progress.bar import Bar
from PIL import Image
import numpy as np
import socket

parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='test',
                        help='save name for model')
parser.add_argument('--project_name', type=str, default='test',
                        help='Name for wandb project')
parser.add_argument('--run_name', type=str, default='test',
                        help='Name for wandb run')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--wandb_MODE', type=str, default='online',
                        help='wandb_MODE for running offline')
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
parser.add_argument('--custom_pretrained_weights', type=str, default=None,
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


    # hostname = str(torch.cuda.current_device()) + socket.gethostname()
    working_dir = os.path.join(args.root, args.data_dir, wandb.run.name)
    # Ensure the working directory is empty
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # Define source directories
    src_dir_easy = os.path.join(args.root, "dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Easy")
    # src_dir_difficult = os.path.join(args.root, "dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Difficult")
    # src_dir_unsure = os.path.join(args.root, "dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Unsure")

    # Define new subdirectories in the working directory
    dest_dir_easy = os.path.join(working_dir, "Easy")
    dest_dir_difficult = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Difficult"
    dest_dir_unsure = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500/Unsure"

    # Copy the contents of the source directories to the new subdirectories
    shutil.copytree(src_dir_easy, dest_dir_easy)
    # shutil.copytree(src_dir_difficult, dest_dir_difficult)
    # shutil.copytree(src_dir_unsure, dest_dir_unsure)

    classes = sorted(os.listdir(os.path.join(working_dir, "Easy/val")))

    # Add a dumby read only image to each class file for the dificult images because when relabled 
    # the directory can become empty thus causeing an error from the dataloader
    # for class_ in classes:
    #     # Define the path for the dummy JPEG file
    #     dif_dummy_file_path = os.path.join(working_dir, "Difficult/val", class_, "dummy.jpeg")
    #     uns_dummy_file_path = os.path.join(working_dir, "Unsure/val", class_, "dummy.jpeg")

    #     # Create a valid dummy JPEG image
    #     dummy_image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))  # Creates a 10x10 black image
    #     dummy_image.save(dif_dummy_file_path, "JPEG")
    #     dummy_image.save(uns_dummy_file_path, "JPEG")

    #     # Set the file to read-only state
    #     os.chmod(dif_dummy_file_path, 0o444)  # Read-only permissions
    #     os.chmod(uns_dummy_file_path, 0o444)  # Read-only permissions

    num_classes = len(os.listdir(dest_dir_easy + "/train"))
    # data_dir, num_classes, _ = toolbox.setup(args)
    device = torch.device("cuda:" + args.GPU)



    #define config dictionary with wandb
    config = {
        'input_size': wandb.config.input_size,
        'dim_1': wandb.config.dim_1, 
        'dim_2': wandb.config.dim_2, 
        'dim_3': wandb.config.dim_3,
        'kernel_1': wandb.config.kernel_1, 
        'kernel_2': wandb.config.kernel_2,
        'kernel_3': wandb.config.kernel_3,
        'num_blocks_1': wandb.config.num_blocks_1,
        'num_blocks_2': wandb.config.num_blocks_2,
        'out_channels': wandb.config.out_channels,    
        'batch_size': args.batch_size,
        'beta1': wandb.config.beta1,
        'beta2': wandb.config.beta2,  
        'learning_rate': wandb.config.learning_rate,
    }
    # config = {    
    #     'DFLoss_delta': 0.06379802231720144,
    #     'beta1': 0.9,
    #     'beta2': 0.943630021404608,
    #     'dim_1': 116,
    #     'dim_2': 106,
    #     'dim_3': 84, 
    #     'input_size': 240, 
    #     'kernel_1': 3,
    #     'kernel_2': 5,
    #     'kernel_3': 13,
    #     'learning_rate': 0.0001,
    #     'num_blocks_1': 4,
    #     'num_blocks_2': 1,
    #     'out_channels': int(num_classes*1.396007582340178),
    #     'batch_size': args.batch_size
    # }
    
    model = toolbox.build_model(arch=args.arch, num_classes=None, config=config).to(device)

    #load weights
    if args.custom_pretrained_weights != None:
        PhyloNetWeights = torch.load(os.path.join(args.root, args.custom_pretrained_weights), map_location=device)
        PhyloNetWeights = {k: v for k, v in PhyloNetWeights.items() if 'fc' not in k}
        # Load the modified state dict
        model.load_state_dict(PhyloNetWeights, strict=False)
        print('\nLoaded weights from: ', args.custom_pretrained_weights)


    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                        weight_decay=args.weight_decay, eps=args.eps, betas=(config['beta1'],config['beta2']))
        

   
    input_size = torch.Size([3, config['input_size'], config['input_size']])
    
    GFLOPs, n_params = toolbox.count_flops(model=model, device=device, input_size=input_size)
    wandb.log({'GFLOPs': GFLOPs, 'n_params': n_params})  # Log the GFLOPs and n_params of the model
  
    print()
    print('GFLOPs: ', GFLOPs, 'n_params: ', n_params)

    if GFLOPs < 6 and n_params < 5000000:
        for major_epoch, data_set in enumerate([dest_dir_difficult, dest_dir_unsure]):
            minor_epoch = 0
            n_relabled = 1
            # n_to_relable = 0
            # for _, _, files in os.walk(os.path.join(data_set, "val")):
            #     n_to_relable += len(files)
            # n_to_relable = 5 #4 is the minimum value does to the 4 dumby images added to each class directory
            while n_relabled > 0:# and n_to_relable > 4:
                print("\nMajor epoch: ", major_epoch, ":", minor_epoch)
                minor_epoch += 1

                image_datasets = toolbox.build_datasets(data_dir=dest_dir_easy, input_size=config['input_size']) #If images are pre compressed, use input_size=None, else use input_size=args.input_size
                dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=6, worker_init_fn=toolbox.worker_init_fn, drop_last=False) for x in ['train', 'val']}
                model_out, best_f1, best_f1_loss, best_train_f1, best_train_metrics, best_val_metrics = train_model(args=args, 
                                                                                                  model=model, 
                                                                                                  optimizer=optimizer, 
                                                                                                  device=device, 
                                                                                                  dataloaders_dict=dataloaders_dict, 
                                                                                                  criterion=criterion, 
                                                                                                  patience=args.patience, 
                                                                                                  batch_size=args.batch_size,
                                                                                                  num_classes=num_classes)
                print("\nRelabeling images")
                model_out.eval()
                dif_datasets = toolbox.build_datasets(data_dir=data_set, input_size=config['input_size'])
                dif_dataloader = torch.utils.data.DataLoader(dif_datasets['val'], batch_size=1, shuffle=False, num_workers=6, worker_init_fn=toolbox.worker_init_fn, drop_last=False)

                n_relabled = 0
                for idx, (image, label) in enumerate(dif_dataloader):
                    _, _, pred = model_out(image.to(device))
                    pred = torch.argmax(pred, dim=1)
                    if pred.item() == label.to(device).item():
                        image_path = dif_datasets['val'].imgs[idx][0]
                        dest = os.path.join(working_dir, "Easy/train", classes[label.item()], os.path.basename(image_path))
                        if os.path.exists(dest) == False:
                            shutil.copy(image_path, dest)
                            n_relabled += 1

                # n_to_relable = 0
                # for _, _, files in os.walk(os.path.join(data_set, "val")):
                #     n_to_relable += len(files)

                print("\nRelabeled: ", n_relabled, " images")
                # print("\nTo relabel: ", n_to_relable-n_relabled, " images")
        
    else: 
        print()
        print('Model too large, aborting training')
        print()
        run.log({'Status': 'aborted'})  # Log the status as 'aborted'
        run.finish()  # Finish the run

        model_out, best_f1, best_f1_loss, best_train_f1, config = None, None, None, None, None
    
    wandb.finish()
    shutil.rmtree(working_dir)
    return model_out, best_f1, best_f1_loss, best_train_f1, config

os.environ["wandb__SERVICE_WAIT"] = "300"
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