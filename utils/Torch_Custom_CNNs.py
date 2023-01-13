from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import copy
import wandb
import pprint
import random
import pickle
from sklearn import metrics
from progress.bar import Bar
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, ResNet50_Weights, ResNeXt101_32X8D_Weights
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse
import yaml

#Local scripts
from ConvNext_tiny_quantised import convnext_tiny as convnext_tiny_q
from ConvNext_tiny_quantised import ConvNeXt_Tiny_Weights as ConvNeXt_Tiny_Weights_q

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
                        help='Initial batch size')

parser.add_argument('--root', type=str, default='/local/scratch/jrs596/dat',
                        help='location of all data')
parser.add_argument('--data_dir', type=str, default='FAIGB_Combined_FinalSplit',
                        help='location of all data')

parser.add_argument('--custom_pretrained', action='store_true', default=False,
                        help='Train useing specified pre-trained weights?')
parser.add_argument('--custom_pretrained_weights', type=str,
                        help='location of pre-trained weights')

parser.add_argument('--quantise', action='store_true', default=False,
                        help='Train with Quantization Aware Training?')

parser.add_argument('--initial_batch_size', type=int, default=32,
                        help='Initial batch size')
parser.add_argument('--initial_num_classes', type=int, default=2,
                        help='Initial number of classes to start traning')
parser.add_argument('--min_batch_size', type=int, default=4,
                        help='Minimum batch size before training ends')
parser.add_argument('--max_epochs', type=int, default=500,
                        help='n epochs before early stopping')
parser.add_argument('--min_epochs', type=int, default=10,
                        help='n epochs before loss is assesed for early stopping')
parser.add_argument('--patience', type=int, default=50,
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
parser.add_argument('--subset_classes_balance', action='store_true', default=False,
                        help='When loss stops decreasing, increase n classes by one and re-subsample the most common classes to match the least frequent?')
parser.add_argument('--remove_batch_norm', action='store_true', default=False,
                        help='Deactivate all batchnorm layers?')


args = parser.parse_args()
print(args)

#Define some variable and paths
os.environ['TORCH_HOME'] = os.path.join(args.root, "TORCH_HOME")
data_dir = os.path.join(args.root, args.data_dir)
model_path = os.path.join(args.root, 'models')
log_dir= os.path.join(model_path, "logs", "logs_" + args.model_name)

if args.subset_classes_balance == False:
    num_classes = len(os.listdir(os.path.join(data_dir, 'val')))
elif args.swsubset_classes_balanceeep == True:
    num_classes=args.initial_num_classes

num_workers = os.cpu_count()-2


#Set environment variables for wandb sweep
os.environ['WANDB_CACHE_DIR'] = os.path.join(args.root, 'WANDB_CACHE')
os.environ['WANDB_DIR'] = os.path.join(args.root, 'WANDB_DIR')

if args.sweep:
    with open(args.sweep_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    sweep_config = config['sweep_config']
    metric = config['metric']
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


def train_model(model, optimizer, image_datasets, criterion, patience, initial_bias, num_classes):   
    since = time.time()
    val_loss_history = []
    best_f1 = 0.0
    best_f1_acc = 0.0

    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())
    

    #######################################################################
    #Initialise dataloader and set decaying batch size
    batch_size = args.initial_batch_size
    
    if args.subset_classes_balance == True:
        #Incrementally increase the number of classes used in training and balance the dataset with continuoiuse resampling
        dataloaders_dict, num_classes, patience, val_loss_history = subset_classes_balance(num_classes=num_classes, batch_size=batch_size, patience=patience, val_loss_history=val_loss_history)
    else:
        #Add initial bias to last layer
        best_model_wts['module.fc.bias'] = initial_bias
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}

    epoch = 0
    #while patience >= 0: # Run untill validation loss has not improved for n epochs equal to patience variable and batchsize has decaed to 1
    while patience >= 0 and epoch < args.max_epochs:
        print('\nEpoch {}'.format(epoch))
        print('-' * 10) 

        #Ensure minimum number of epochs is met before patience is allow to reduce
        if len(val_loss_history) > args.min_epochs:
           #If the current loss is not at least 0.5% less than the lowest loss recorded, reduce patiece by one epoch
            if val_loss_history[-1] >= min(val_loss_history)*args.beta:
                patience -= 1
            elif val_loss_history[-1] == np.nan:
                patience -= 1
            else:
               #If validation loss improves by at least 0.5%, reset patient to initial value
               patience = args.patience
        print('Patience: ' + str(patience) + '/' + str(args.patience))
       #If patience gets to 6, half batch size, reintialise dataloader and revert to best model weights to undo any overfitting.
       #Do this for every epoch where patience is less than 6. i.e. if inital batch size = 128, the last epoch will have a batchsize of at most 2.
        if patience < 6 and batch_size > args.min_batch_size:
            batch_size = int(batch_size/2)
            print('Batch size: ' + str(batch_size))
            
        if patience <= 40 and patience % 10 == 0:
           #Re-initialise dataloaders and re-randomise batches.
             
            if args.subset_classes_balance == True:
                #Increase number of classes used in training and balance the dataset with continuoiuse resampling
                
                #patience = args.patience
                batch_size = args.initial_batch_size
                
                dataloaders_dict, num_classes, patience, val_loss_history = subset_classes_balance(num_classes=num_classes, batch_size=batch_size, patience=patience, val_loss_history=val_loss_history)
                patience = patience - 1 
                print("Current number of classes: ", str(num_classes))

                #Re-initialise last layer of model with new number of classes
                #Revert to best model weights so far
                if args.arch == 'resnet18' or args.arch == 'resnet50':
                    in_feat = model.module.fc.in_features
                    model.module.fc = nn.Linear(in_feat, num_classes)
                    best_model_wts['module.fc.weight'] = torch.rand(num_classes, in_feat)
                    best_model_wts['module.fc.bias'] = torch.rand(num_classes)
                elif args.arch == 'convnext_tiny':
                    #TODO: Make sure this works for convnext
                    in_feat = model.classifier[2].in_features
                    model.classifier[2] = torch.nn.Linear(in_feat, num_classes)
                    best_model_wts['classifier.2.weight'] = torch.rand(num_classes, in_feat)
                    best_model_wts['classifier.2.bias'] = torch.rand(num_classes)
                
                model.load_state_dict(best_model_wts)
                model.to(device)

                #Reset best f1 score to 0 so that the model is saved even if the new number of classes worsens the f1 score
                best_f1 = 0.0
                best_f1_acc = 0.0
            else:
                dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}

           #Reload best model weights
            model.load_state_dict(best_model_wts)
        print('\n Batch size: ' , batch_size)
        print(' Number of classes in use: ' + str(num_classes) + '/' + str(len(os.listdir(os.path.join(data_dir, 'val')))) + '\n')
 
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
                        #else: 
                            #if idx == 0:
                                # Log one batch of images to the dashboard, always same batch_idx.
                                #log_image_table(inputs, preds, labels, outputs.softmax(dim=1))
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
            


           # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_f1_acc = epoch_acc
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
   
            if phase == 'train':
                wandb.log({"epoch": epoch, "Train_loss": epoch_loss, "Train_acc": epoch_acc, "Train_F1": epoch_f1})  
            else:
                wandb.log({"epoch": epoch, "Val_loss": epoch_loss, "Val_acc": epoch_acc, "Val_F1": epoch_f1, "Best_F1": best_f1, "Best_F1_acc": best_f1_acc})

        bar.finish() 
        epoch += 1
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Acc of saved model: {:4f}'.format(best_f1_acc))
    print('Recall of saved model: {:4f}'.format(best_f1))
    


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


def build_datasets(kernel_size, sigma_max, input_size, data_dir):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(input_size, pad_if_needed=True, padding_mode = 'reflect'),
            #transforms.Resize((input_size,input_size)),
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


def Remove_module_from_layers(unpickled_model_wts):
    new_keys = []
    for key, value in unpickled_model_wts.items():
        new_keys.append(key.replace('module.', ''))
    for i in new_keys:
        unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)#    
    return unpickled_model_wts

def build_model(num_classes):
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


def subset_classes_balance(num_classes, batch_size, patience, val_loss_history):
    if args.sweep == True:
         image_datasets = build_datasets(kernel_size=wandb.config.kernel_size, sigma_max=wandb.config.sigma_max, input_size=int(wandb.config.input_size), data_dir=data_dir)
    else:
         image_datasets = build_datasets(kernel_size=5, sigma_max=2, input_size=args.input_size, data_dir=data_dir)

    if patience == 0:
        if num_classes < len(os.listdir(os.path.join(data_dir, 'val'))):
            num_classes += 1
            patience = args.patience
            val_loss_history = []
            print("\n########################################")
            print("Shuffeling and balancing dataset and adding a new class")
            print("########################################\n")
        else:
            num_classes = len(os.listdir(os.path.join(data_dir, 'val')))
    else:
        print("\n########################################")
        print("Shuffeling and balancing dataset")
        print("########################################\n")
        
    #list all classes
    classe_names = image_datasets['train'].classes
    print("\nClasses in current training set: ", classe_names[0:num_classes])
    #create list of first n classes
    classes = list(range(len(classe_names[0:num_classes])))

    #create dictionary with key = class and value = indices of images in class
    indices_dict = {}
    for i in classes:
        class_tensor = torch.tensor(i)
        indices = (torch.tensor(image_datasets['train'].targets)[..., None] == class_tensor).any(-1).nonzero(as_tuple=True)[0]
        indices_dict[i] = indices

    #get lenght of shortest key in indices_dict
    min_len = len(indices_dict[0]) 
    for key in indices_dict:
        if len(indices_dict[key]) < min_len:
            min_len = len(indices_dict[key])
    if min_len < 50:
        min_len = 50
    indices = []
    for key in indices_dict:
        #random sample value of size min_len from indices_dict[key]
        if len(indices_dict[key]) > min_len:
            sample = random.sample(list(indices_dict[key]), min_len)
        else:
            sample = list(indices_dict[key])
        indices.append(sample)
    #flatten list of lists
    indices = [item for sublist in indices for item in sublist]
    image_datasets['train'] = torch.utils.data.Subset(image_datasets['train'], indices)

    #subset validation set. Full size but only first n classes
    val_indices = (torch.tensor(image_datasets['val'].targets)[..., None] == torch.tensor(classes)).any(-1).nonzero(as_tuple=True)[0]
    image_datasets['val'] = torch.utils.data.Subset(image_datasets['val'], val_indices)

    print("\nCurrent train subset size: " + str(len(image_datasets['train'])))
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}

    return dataloaders_dict, num_classes, patience, val_loss_history

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


def sweep_train(config=sweep_config):
    # Initialize a new wandb run
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
    run = wandb.init(config=config)

    model_ft = build_model(num_classes=num_classes)
    
    model_ft = set_batchnorm_momentum(model_ft, wandb.config.batchnorm_momentum)
    model_ft = model_ft.to(device)  
    optimizer = build_optimizer(model_ft, wandb.config.optimizer, wandb.config.learning_rate, wandb.config.eps, wandb.config.weight_decay)
    image_datasets = build_datasets(kernel_size=wandb.config.kernel_size, sigma_max=wandb.config.sigma_max, input_size=int(wandb.config.input_size), data_dir=data_dir)

    train_model(model=model_ft, optimizer=optimizer, image_datasets=image_datasets, criterion=criterion, patience=args.patience, initial_bias=initial_bias, num_classes=num_classes)

def train():
    wandb.init(project=args.project_name)
    num_classes = len(os.listdir(os.path.join(data_dir, 'val')))
    model_ft = build_model(num_classes=num_classes)
    model_ft = model_ft.to(device)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-5,
                                           weight_decay=0, eps=1e-8)
    image_datasets = build_datasets(kernel_size=5, sigma_max=2, input_size=args.input_size, data_dir=data_dir)

    train_model(model=model_ft, optimizer=optimizer, image_datasets=image_datasets, criterion=criterion, patience=args.patience, initial_bias=initial_bias, num_classes=num_classes)


if args.sweep == True:
    wandb.agent(sweep_id,
        project=args.project_name, 
        function=sweep_train,
        count=args.sweep_count)
else:
    train()