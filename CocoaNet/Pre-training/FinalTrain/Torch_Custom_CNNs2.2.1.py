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
from collections import OrderedDict

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
parser.add_argument('--save', action='store_true', default=False,
                        help='save model weigths ?')
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
parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Log data with wandb?')


args = parser.parse_args()
print(args)

sys.path.append(os.path.join(os.getcwd(), 'scripts/CocoaReader/utils'))
import toolbox
from training_loop import train_model

def Relabel(model, device, data_dir, working_dir, copied_images, classes, config):
    print("\nRelabeling images from:\n ", data_dir, "\n")
    model.eval()
    
    # Build datasets and dataloader for the current data_set (dest_dir_difficult or dest_dir_unsure)
    dif_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=config['input_size'])
    dif_dataloader = torch.utils.data.DataLoader(dif_datasets['val'], batch_size=200, shuffle=False, num_workers=6, worker_init_fn=toolbox.worker_init_fn, drop_last=False)
    n_relabeled = 0
    
    # Iterate over the images in the dataloader and relabel them if the model's prediction matches the ground truth label
    for idx, (images, labels) in enumerate(dif_dataloader):
        _, _, preds = model(images.to(device))
        preds = torch.argmax(preds, dim=1)
        for i in range(images.size(0)):  # Loop through each item in the batch
            #Healthy = 2, NotCocoa = 3
            #There are not difficult or unsure Healthy of NotCocoa images, just dumby images
            included_labels = [0, 1, 4]
            if labels[i].item() in included_labels:
                if preds[i].item() == labels[i].to(device).item():
                    image_path = dif_datasets['val'].imgs[idx * dif_dataloader.batch_size + i][0]
                    label = labels[i].item()
                    dest = os.path.join(working_dir, "Easy/train", classes[label], os.path.basename(image_path))

                    # Copy the image to the destination directory if it hasn't been copied before
                    if image_path not in copied_images and not os.path.exists(dest):
                        shutil.copy(image_path, dest)
                        copied_images.add(image_path)
                        n_relabeled += 1
    print("\nRelabeled: ", n_relabeled, " images")
    


def train():

    if args.use_wandb:
        run = wandb.init(project=args.project_name, settings=wandb.Settings(_service_wait=300))
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        wandb.save(os.path.join(script_dir, '*'), base_path=script_dir) 
        working_dir = os.path.join(args.root, args.data_dir, "Working_Dir", wandb.run.name)    
    else:
        working_dir = os.path.join(args.root, args.data_dir, "Working_Dir", args.run_name)


    #Set seeds for reproducability
    toolbox.SetSeeds(42)       
    # Ensure the working directory is empty
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # Define source directories
    src_dir_easy = os.path.join(args.root, args.data_dir, "Easy")

    # Define new subdirectories in the working directory
    dest_dir_easy = os.path.join(working_dir, "Easy")
    dest_dir_difficult = os.path.join(args.root, args.data_dir, "Difficult")
    dest_dir_unsure = os.path.join(args.root, args.data_dir, "Unsure")

    # Copy the contents of the source directories to the new subdirectories
    shutil.copytree(src_dir_easy, dest_dir_easy)

    classes = sorted(os.listdir(os.path.join(working_dir, "Easy", "val")))
    num_classes = len(classes)
        
    # device = torch.device("cuda:" + args.GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = {
            'beta1': 0.9305889820653824,  
            'beta2': 0.977926878163776,  
            'dim_1': 125,  
            'dim_2': 80,  
            'dim_3': 54, 
            'input_size': args.input_size,  
            'kernel_1': 3,  
            'kernel_2': 11,  
            'kernel_3': 17,  
            'learning_rate': 0.0007653560770141792,  
            'num_blocks_1': 3,  
            'num_blocks_2': 2,  
            'out_channels': num_classes,  
            'num_heads': 3, #3 for PhytNetV0, 4 for ResNet18  
            'batch_size': args.batch_size,  
            'num_decoder_layers': 4,
        }
    
    # config = {
    #         'beta1': wandb.config.beta1,  
    #         'beta2': wandb.config.beta2, 
    #         'dim_1': wandb.config.dim_1,
    #         'dim_2': wandb.config.dim_2,
    #         'dim_3': wandb.config.dim_3, 
    #         'input_size': wandb.config.input_size,  
    #         'kernel_1': wandb.config.kernel_1,  
    #         'kernel_2': wandb.config.kernel_2,
    #         'kernel_3': wandb.config.kernel_3,
    #         'learning_rate': wandb.config.learning_rate,  
    #         'num_blocks_1': wandb.config.num_blocks_1,
    #         'num_blocks_2': wandb.config.num_blocks_2,
    #         'out_channels': wandb.config.out_channels,  
    #         'num_heads': 3, #3 for PhytNetV0, 4 for ResNet18  
    #         'batch_size': args.batch_size,  
    #         'num_decoder_layers': 4,
    #     }


    model = toolbox.build_model(arch=args.arch, num_classes=config['out_channels'], config=config).to(device)
    
    # #For max performance on H100 GPUs
    # torch.set_float32_matmul_precision('high')
    # model = torch.compile(model)
    
    if args.custom_pretrained_weights != None:
        PhytNetWeights = torch.load(os.path.join(args.root, args.custom_pretrained_weights), map_location=device)      
        
        # try:
        #     # Load the entire state dictionary
        #     PhytNetWeights = OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in PhytNetWeights.items())

        #     # Filter out encoder keys and values
        #     encoder_prefix = 'encoder.'  # Adjust this prefix based on the actual naming convention
        #     PhytNetWeights = OrderedDict((k[len(encoder_prefix):], v) for k, v in PhytNetWeights.items() if k.startswith(encoder_prefix))
       

        #     #Assign new fc layer with num_classes output channels
        PhytNetWeights['fc.weight'] = torch.rand((num_classes, PhytNetWeights['fc.weight'].shape[1]), dtype=PhytNetWeights['fc.weight'].dtype, device=PhytNetWeights['fc.weight'].device)
        PhytNetWeights['fc.bias'] = torch.rand((num_classes,), dtype=PhytNetWeights['fc.bias'].dtype, device=PhytNetWeights['fc.bias'].device)
        
        # except:
        #     pass
        
    
        # Assuming your model's encoder is accessible as `model.encoder`
        # Load the encoder weights
        model.load_state_dict(PhytNetWeights, strict=True)  # Use strict=False if the encoder's parameters don't exactly match
        
        # model.load_state_dict(PhytNetWeights, strict=True)
        print('\nLoaded weights from: ', args.custom_pretrained_weights)
    else:
        print('\nNo weights loaded')

    # criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                        weight_decay=args.weight_decay, eps=args.eps, betas=(config['beta1'], config['beta2']))
        
    prev_best_f1 = 0.0
    copied_images = set()

    # Loop through two different datasets: dest_dir_difficult and dest_dir_unsure
    for major_epoch, data_dir in enumerate([dest_dir_difficult, dest_dir_unsure]):
        minor_epoch = 0
        n_relabeled = 1

        if major_epoch == 1:
            Relabel(model=model, device=device, data_dir=data_dir, working_dir=working_dir, copied_images=copied_images, classes=classes, config=config)

        while n_relabeled > 0:
            print("\nMajor epoch: ", major_epoch, ":", minor_epoch)
            minor_epoch += 1
            
            # Build datasets and dataloaders for the easy dataset
            
            image_datasets = toolbox.build_datasets(data_dir=dest_dir_easy, input_size=config['input_size'])
            dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=6, worker_init_fn=toolbox.worker_init_fn, drop_last=False) for x in ['train', 'val']}
            criterion = toolbox.DynamicFocalLoss(delta=3.7344 , dataloader=dataloaders_dict['train'])
            
 
            # Train the model using the train_model function
            model, best_f1 = train_model(args=args, 
                                         model=model, 
                                         optimizer=optimizer, 
                                         device=device, 
                                         dataloaders_dict=dataloaders_dict, 
                                         criterion=criterion, 
                                         patience=args.patience, 
                                         batch_size=args.batch_size,
                                         num_classes=config['out_channels'],
                                         best_f1=prev_best_f1)         
            
            if best_f1 <= prev_best_f1:
                break
            else:
                prev_best_f1 = best_f1  
            
                Relabel(model=model, device=device, data_dir=data_dir, working_dir=working_dir, copied_images=copied_images, classes=classes, config=config)

           

    wandb.finish()
    shutil.rmtree(working_dir)

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