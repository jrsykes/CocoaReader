from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
#from torch.utils.tensorboard import SummaryWriter
import wandb
import pprint

import pickle
from sklearn import metrics
from progress.bar import Bar
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, ResNet50_Weights, ResNeXt101_32X8D_Weights
from torch.utils.mobile_optimizer import optimize_for_mobile

import sys
import argparse

#Local scripts
from ConvNext_tiny_quantised import convnext_tiny as convnext_tiny_q
from ConvNext_tiny_quantised import ConvNeXt_Tiny_Weights as ConvNeXt_Tiny_Weights_q

parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str,
                        help='save name for model')

parser.add_argument('--sweep', action='store_true',
                        help='Run Waits and Biases optimisation sweep')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--sweep_count', type=int, default=4,
                        help='Initial batch size')

parser.add_argument('--root', type=str,
                        help='location of all data')
parser.add_argument('--data_dir', type=str,
                        help='location of all data')

parser.add_argument('--custom_pretrained', action='store_true',
                        help='Train useing specified pre-trained weights?')
parser.add_argument('--custom_pretrained_weights', type=str,
                        help='location of pre-trained weights')

parser.add_argument('--quantise', action='store_true',
                        help='Train with Quantization Aware Training?')

parser.add_argument('--initial_batch_size', type=int, default=128,
                        help='Initial batch size')
parser.add_argument('--min_batch_size', type=int, default=8,
                        help='Minimum batch size before training ends')
parser.add_argument('--min_epochs', type=int, default=10,
                        help='n epochs before loss is assesed for early stopping')
parser.add_argument('--patience', type=int, default=50,
                        help='n epochs to run without improvment in loss')
parser.add_argument('--beta', type=float, default=1,
                        help='minimum required per cent improvment in validation loss')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps, Default:1e-8')
parser.add_argument('--input_size', type=int, default=224,
                        help='image input size')
parser.add_argument('--arch', type=str, default='resnet18',
                        help='Model architecture. resnet18, resnet50, resnext50, resnext101 or convnext_tiny')
parser.add_argument('--cont_train', action='store_true',
                        help='Continue training from previous checkpoint?')
parser.add_argument('--remove_batch_norm', action='store_true',
                        help='Deactivate all batchnorm layers?')



args = parser.parse_args()
print(args)

#Define some variable and paths
data_dir = os.path.join(args.root, args.data_dir)
model_path = os.path.join(args.root, 'models')
log_dir= os.path.join(model_path, "logs", "logs_" + args.model_name)
num_classes = len(os.listdir(os.path.join(data_dir, 'val')))
#writer = SummaryWriter(log_dir=log_dir)

sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'Val_F1',
    'goal': 'maximize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'input_size': {
        'distribution': 'uniform',
        'min': 448,
        'max': 1120
      },
    'kernel_size': {
        'values': [3,5,7,9,11]
      },
    'learning_rate': {
        # a flat distribution between min and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'eps': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'weight_decay': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'sigma_max': {
        'distribution': 'uniform',
        'min': 1,
        'max': 5
      },
    'batchnorm_momentum': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 2
      }
    }


sweep_config['parameters'] = parameters_dict
print('Sweep config:')
pprint.pprint(sweep_config)
print()


if args.sweep_id is None:
    sweep_id = wandb.sweep(sweep_config, project="DisNet", entity="frankslab")
else:
    sweep_id = args.sweep_id

print("Sweep ID: ", sweep_id)
print()


def train_model(model, optimizer, image_datasets, criterion, patience, initial_bias):
    since = time.time()
    val_loss_history = []
    best_recall = 0.0
    best_recall_acc = 0.0
    
    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())
    #Add initial bias to last layer
    best_model_wts['module.fc.bias'] = initial_bias
    #######################################################################
    #Initialise dataloader and set decaying batch size
    batch_size = args.initial_batch_size
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=12) for x in ['train', 'val']}    
    #initial_patience = patience
    epoch = 0
    while batch_size >= args.min_batch_size: # Run untill validation loss has not improved for n epochs equal to patience variable and batchsize has decaed to 1
        print('\nEpoch {}'.format(epoch))
        print('-' * 10) 
        #Ensure minimum number of epochs is met before patience is allow to reduce
        if len(val_loss_history) > args.min_epochs:
           #If the current loss is not at least 0.5% less than the lowest loss recorded, reduce patiece by one epoch
            if val_loss_history[-1] > min(val_loss_history)*args.beta:
                patience -= 1
            else:
               #If validation loss improves by at least 0.5%, reset patient to initial value
               patience = args.patience
        print('Patience: ' + str(patience) + '/' + str(args.patience))
       
       #If patience gets to 6, half batch size, reintialise dataloader and revert to best model weights to undo any overfitting.
       #Do this for every epoch where patience is less than 6. i.e. if inital batch size = 128, the last epoch will have a batchsize of at most 2.
        if patience == 0:
            batch_size = int(batch_size/2)
            patience = args.patience
           #Re-initialise dataloaders and rerandomise batches.  
            dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=12) for x in ['train', 'val']}
           #Reload best model weights
            model.load_state_dict(best_model_wts)
        print('\n Batch size: ' , batch_size, '\n') 
       #Training and validation loop
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
               #Experimental. Remove batch norm layer from Resnet if specified.
                if args.remove_batch_norm == True:
                    print('BatchNorm layers deactivated')
                    model.apply(deactivate_batchnorm)
            elif phase == 'val':
               #Model quatisation with quantisation aware training
                if args.quantise == True:
                    quantized_model = torch.quantization.convert(model.eval(), inplace=False)
                    quantized_model.eval()
                model.eval()   # Set model to evaluate mode
               
            running_loss = 0.0
            running_corrects = 0
            running_precision = 0
            running_recall = 0
            running_f1 = 0  
           # Iterate over data.
           #Get size of whole dataset split
            n = len(dataloaders_dict[phase].dataset)
           #Begin training
            print(phase)
           #n_steps_per_epoch = len(dataloaders_dict['train'].dataset)/batch_size
            with Bar('Learning...', max=n/batch_size+1) as bar:
               
                for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
                    #Load images and lables from current batch onto GPU(s)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()       
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                       # Get model outputs and calculate loss
                       # In train mode we calculate the loss by summing the final output and the auxiliary output
                       # but in testing we only consider the final output.
                       
                       #Get predictionas from regular or quantised model
                        if args.quantise == True and phase == 'val':
                            outputs = quantized_model(inputs)
                        else:
                            outputs = model(inputs)
                       
                       #Calculate loss and other model metrics
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)    
                        stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
                        stats_out = stats['weighted avg']
                       
                       #Add precision, recall or f1-score to loss with weight. Experimental. Doesn't seem to work in practice.
                       #loss += (1-stats_out['recall'])#*0.4   
                       # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()    
                            if args.quantise == True:
                                if epoch > 3:
                                    model.apply(torch.quantization.disable_observer)
                                if epoch > 2:
                                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                        else: 
                            if idx == 0:
                                # Log one batch of images to the dashboard, always same batch_idx.
                                log_image_table(inputs, preds, labels, outputs.softmax(dim=1))
                   #Calculate statistics
                   #Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
                   #running totals for these metrics by the total number of training or validation samples. This controls for 
                   #the effect of batch size and the fact that the size of the last batch will less than args.batch_size
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data) 
                    running_precision += stats_out['precision'] * inputs.size(0)
                    running_recall += stats_out['recall'] * inputs.size(0)
                    running_f1 += stats_out['f1-score'] * inputs.size(0)    

                    bar.next()  
           #Calculate statistics for epoch
            n = len(dataloaders_dict[phase].dataset)
            epoch_loss = float(running_loss / n)
            epoch_acc = float(running_corrects.double() / n)
            epoch_precision = (running_precision) / n         
            epoch_recall = (running_recall) / n        
            epoch_f1 = (running_f1) / n 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_precision, epoch_recall, epoch_f1))
            # Save statistics to tensorboard log
            
            if phase == 'train':
                wandb.log({"epoch": epoch, "Train_loss": epoch_loss, "Train_acc": epoch_acc, "Train_F1": epoch_f1})  
            else:
                wandb.log({"epoch": epoch, "Val_loss": epoch_loss, "Val_acc": epoch_acc, "Val_F1": epoch_f1})  

           # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_recall > best_recall:
                best_recall = epoch_recall
                best_recall_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  
               # Save only the model weights for easy loading into a new model
                final_out = {
                    'model': best_model_wts,
                    '__author__': 'Jamie R. Sykes',
                    '__model_name__': args.model_name,
                    '__model_parameters__': args                    
                    }       
                
                PATH = os.path.join(model_path, args.model_name)
                   
                if args.quantise == False:
                    with open(PATH + '.pkl', 'wb') as f:
                        pickle.dump(final_out, f)       
                   # Save the whole model with pytorch save function
                    torch.save(model.module, PATH + '.pth') 
                else:
                   # Convert the quantized model to torchscipt and optmize for mobile platforms
                    torchscript_model = torch.jit.script(quantized_model)
                    torch.jit.save(torchscript_model, PATH + '.pth')
                    optimized_torchscript_model = optimize_for_mobile(torchscript_model)
                    optimized_torchscript_model.save(PATH + "_mobile.pth")
                    optimized_torchscript_model._save_for_lite_interpreter(PATH + "_mobile.ptl")    
            if phase == 'val':
                val_loss_history.append(epoch_loss)
    
        epoch += 1
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Acc of saved model: {:4f}'.format(best_recall_acc))
    print('Recall of saved model: {:4f}'.format(best_recall))
    


def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"class_{i}" for i in range(num_classes)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)

def build_optimizer(network, optimizer, learning_rate, eps, weight_decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay, eps=eps)
    return optimizer


def build_datasets(kernel_size, sigma_max, input_size):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(input_size, pad_if_needed=True, padding_mode = 'reflect'),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, sigma_max)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
        ]),
    }   

    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    return image_datasets

# Specify whether to use GPU or cpu. Quantisation aware training is not yet avalable for GPU.
if args.quantise == True:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")


#Depriated. Remove later
def Remove_module_from_layers(weights):
    new_keys = []
    for key, value in unpickled_model_wts.items():
        new_keys.append(key.replace('module.', ''))
    for i in new_keys:
        unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)#    
    return unpickled_model_wts

def build_model():
    #If checkpoint weights file exists, load those weights.
    if args.cont_train == True and os.path.exists(os.path.join(model_path, args.model_name + '.pkl')) == True:
        print('Loading checkpoint weights')
        pretrained_model_wts = pickle.load(open(os.path.join(model_path, args.model_name + '.pkl'), "rb"))
        unpickled_model_wts = copy.deepcopy(pretrained_model_wts['model'])  

        unpickled_model_wts = Remove_module_from_layers(unpickled_model_wts)    

        model_ft.load_state_dict(unpickled_model_wts)

    #Chose which model architecture to use and whether to load ImageNet weights or custom weights
    elif args.custom_pretrained == False:
        if args.arch == 'convnext_tiny':
            print('Loaded ConvNext Tiny with pretrained IN weights')
            model_ft = models.convnext_tiny(weights = ConvNeXt_Tiny_Weights.DEFAULT)
            in_feat = model_ft.classifier[2].in_features
            model_ft.classifier[2] = torch.nn.Linear(in_feat, num_classes)
        elif args.arch == 'resnet18':
            print('Loaded ResNet18 with pretrained IN weights')
            model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            in_feat = model_ft.fc.in_features
            model_ft.fc = nn.Linear(in_feat, num_classes)
        elif args.arch == 'resnet50':
            print('Loaded ResNet50 with pretrained IN weights')
            model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            in_feat = model_ft.fc.in_features
            model_ft.fc = nn.Linear(in_feat, num_classes)
        elif args.arch == 'resnext50':
            print('Loaded ResNext50 with pretrained IN weights')
            model_ft = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            in_feat = model_ft.fc.in_features
            model_ft.fc = nn.Linear(in_feat, num_classes)
        elif args.arch == 'resnext101':
            print('Loaded ResNext101 with pretrained IN weights')
            model_ft = models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
            in_feat = model_ft.fc.in_features
            model_ft.fc = nn.Linear(in_feat, num_classes)
        else:
            print("Architecture name not recognised")
            exit(0)
    # Load custom pretrained weights    

    else:
        print('\nLoading custom pre-trained weights with: ')
        pretrained_model_wts = pickle.load(open(os.path.join(model_path, args.custom_pretrained_weights), "rb"))
        unpickled_model_wts = copy.deepcopy(pretrained_model_wts['model'])
        unpickled_model_wts = Remove_module_from_layers(unpickled_model_wts)
        
        if args.arch == 'convnext_tiny':
            print('\tConvNeXt tiny architecture\n')
            if args.quantise == False:
                model_ft = models.convnext_tiny(weights= None)
            else:
                model_ft = convnext_tiny_q(weights = None)
            out_feat = unpickled_model_wts['classifier.2.weight'].size()[0]
            in_feat = model_ft.classifier[2].in_features
            model_ft.classifier[2] = torch.nn.Linear(in_feat, out_feat)
            #Load custom weights
            model_ft.load_state_dict(unpickled_model_wts)
            #Delete final linear layer and replace to match n classes in the dataset
            model_ft.classifier[2] = torch.nn.Linear(in_feat, num_classes)
            
        elif args.arch == 'resnet18':
            print('\tResnet18 architecture\n')
            if args.quantise == False:
                model_ft = models.resnet18(weights= None)
            else:
                model_ft = models.quantization.resnet18(weights=None)   

            in_feat = model_ft.fc.in_features
            out_feat = unpickled_model_wts['fc.weight'].size()[0]
            model_ft.fc = nn.Linear(in_feat, out_feat)
            #Load custom weights
            model_ft.load_state_dict(unpickled_model_wts)
            #Delete final linear layer and replace to match n classes in the dataset
            model_ft.fc = torch.nn.Linear(in_feat, num_classes)
        
        elif args.arch == 'resnet50':
            print('\tResnet50 architecture\n')
            if args.quantise == False:
                model_ft = models.resnet50(weights= None)
            else:
                model_ft = models.quantization.resnet50(weights=None)   

            in_feat = model_ft.fc.in_features
            out_feat = unpickled_model_wts['fc.weight'].size()[0]
            model_ft.fc = nn.Linear(in_feat, out_feat)
            #Load custom weights
            model_ft.load_state_dict(unpickled_model_wts)
            #Delete final linear layer and replace to match n classes in the dataset
            model_ft.fc = torch.nn.Linear(in_feat, num_classes) 
               
        elif args.arch == 'resnext50':
            print('\tResNext50 architecture\n')
            if args.quantise == False:
                model_ft = models.resnext50_32x4d(weights= None)
            else:
                model_ft = resnext50_q(weights=None)    

            in_feat = model_ft.fc.in_features
            out_feat = unpickled_model_wts['fc.weight'].size()[0]
            model_ft.fc = nn.Linear(in_feat, out_feat)
            #Load custom weights
            model_ft.load_state_dict(unpickled_model_wts)
            #Delete final linear layer and replace to match n classes in the dataset
            model_ft.fc = torch.nn.Linear(in_feat, num_classes)
        
        elif args.arch == 'resnext101':
            print('\tResNext101 architecture\n')
            if args.quantise == False:
                model_ft = models.resnext101_32x8d(weights= None)
            else:
                model_ft = models.quantization.resnext101_32x8d(weights=None)   

            in_feat = model_ft.fc.in_features
            out_feat = unpickled_model_wts['fc.weight'].size()[0]
            model_ft.fc = nn.Linear(in_feat, out_feat)
            #Load custom weights
            model_ft.load_state_dict(unpickled_model_wts)
            #Delete final linear layer and replace to match n classes in the dataset
            model_ft.fc = torch.nn.Linear(in_feat, num_classes)
       
        else:
            print("Architecture name not recognised")
            exit(0) 
        
    #Run model on all avalable GPUs
    if args.quantise == False:
        model_ft = nn.DataParallel(model_ft)
    else:
        #model_ft.fuse_model()
        model_ft.eval()
        model_ft = torch.quantization.fuse_modules(model_ft, [['conv1', 'bn1', 'relu']])
        model_ft.train()    

    if args.quantise == True:
        print('Training with Quantization Aware Training on CPU')
        model_ft.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(model_ft, inplace=True)

    return model_ft

#Used in model quantisation to fuse layers
def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


#Experimental. Used to deactivate batch normalisation layers in Resnets
def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

def set_batchnorm_momentum(self, momentum):
    for m in self.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = momentum
    return self


### Calculate and set bias for final layer based on imbalance in dataset classes
dir_ = os.path.join(data_dir, 'train')
list_cats = []
for i in sorted(os.listdir(dir_)):
    path, dirs, files = next(os.walk(os.path.join(dir_, i)))
    list_cats.append(len(files))

weights = []
for i in list_cats:
    weights.append(np.log((max(list_cats)/i)))

initial_bias = torch.FloatTensor(weights).to(device)

#Optional, weight loss function.
criterion = nn.CrossEntropyLoss()


#for _ in range(5):
#    # 🐝 initialise a wandb run
#    wandb.init(
#        project="pytorch-intro",
#        config={
#            "min_epochs": argparse.min_epochs,
#            "batch_size": args.initial_batch_size,
#            "lr": random.uniform(1e-4, 1e-1),
#            "eps": random.uniform(1e-10, 1e-5),
#            })
#    
#    # Copy your config 
#    config = wandb.config


    # Train and evaluate

def sweep_train(config=sweep_config):
    # Initialize a new wandb run
    #with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
    run = wandb.init(config=config)
    #config = wandb.config

    model_ft = build_model()
    
    model_ft = set_batchnorm_momentum(model_ft, wandb.config.batchnorm_momentum)
    model_ft = model_ft.to(device)  
    optimizer = build_optimizer(model_ft, wandb.config.optimizer, wandb.config.learning_rate, wandb.config.eps, wandb.config.weight_decay)
    image_datasets = build_datasets(kernel_size=wandb.config.kernel_size, sigma_max=wandb.config.sigma_max, input_size=int(wandb.config.input_size))

    train_model(model=model_ft, optimizer=optimizer, image_datasets=image_datasets, criterion=criterion, patience=args.patience, initial_bias=initial_bias)

def train():
    model_ft = build_model()
    model_ft = model_ft.to(device)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-5,
                                           weight_decay=0, eps=1e-8)
    image_datasets = build_datasets(kernel_size=5, sigma_max=2, input_size=args.input_size)

    train_model(model=model_ft, optimizer=optimizer, image_datasets=image_datasets, criterion=criterion, patience=args.patience, initial_bias=initial_bias)


if args.sweep == True:
    wandb.agent(sweep_id, 
        sweep_train,
        count=args.sweep_count)
else:
    train()