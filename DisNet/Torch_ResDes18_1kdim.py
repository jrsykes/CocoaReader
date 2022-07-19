from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
from sklearn import metrics
from progress.bar import Bar
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights


import sys
import argparse


parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str,
                        help='save name for model')
parser.add_argument('--root', type=str,
                        help='location of all data')
parser.add_argument('--data_dir', type=str,
                        help='location of all data')
parser.add_argument('--batch_size', type=int, default=42,
                        help='Batch size')
parser.add_argument('--min_epochs', type=int, default=10,
                        help='n epochs before loss is assesed for early stopping')
parser.add_argument('--patience', type=int, default=50,
                        help='n epochs to run without improvment in loss')
parser.add_argument('--beta', type=float, default=1.005,
                        help='minimum required per cent improvment in validation loss')
parser.add_argument('--input_size', type=int, default=224,
                        help='image input size')
parser.add_argument('--arch', type=str, default='resnet18',
                        help='Model architecture. resnet18 or convnext_tiny')
parser.add_argument('--cont_train', type=bool, default=False,
                        help='Continue training from previous checkpoint? True or False?')
args = parser.parse_args()


data_dir = os.path.join(args.root, args.data_dir)
model_path = os.path.join(args.root, 'models')
log_dir= os.path.join(model_path, "logs", "logs_" + args.model_name)

# Number of classes in the dataset
num_classes = len(os.listdir(os.path.join(data_dir, 'val')))

writer = SummaryWriter(log_dir=log_dir)

def train_model(model, dataloaders, criterion, optimizer, patience, input_size, initial_bias):
    since = time.time()
    
    val_loss_history = []
    best_precision = 0.0
    best_precision_acc = 0.0
    
    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())
    #Add initial bias to last layer
    best_model_wts['module.fc.bias'] = initial_bias
    #######################################################################
    
    epoch = 0
    initial_patience = patience
    while patience > 0: # Run untill validation loss has not improved for n epochs equal to patience variable
        print('\nEpoch {}'.format(epoch))
        print('-' * 10)

        if len(val_loss_history) > args.min_epochs:
            if val_loss_history[-1] > min(val_loss_history)*args.beta:
                patience -= 1
            else:
                patience = initial_patience
        print('Patience: ' + str(patience) + '/' + str(initial_patience))
        
        # Each epoch has a training and validation phase
        
        for phase in ['train', 'val']:
            #count = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_precision = 0
            running_recall = 0
            running_f1 = 0

            # Iterate over data.
            n = len(dataloaders_dict[phase].dataset)
            print(phase)
            with Bar('Learning...', max=n/args.batch_size+1) as bar:
                
                for inputs, labels in dataloaders[phase]:
                    #count += 1
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
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)    

                        stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
                        stats_out = stats['weighted avg']
                        #Weight loss function by precision, recall or f1
                        #loss += (1-stats_out['precision'])*0.4

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()    

                    #Calculate statistics
                    #Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
                    #running totals for these metrics by the total number of training or test samples. This controls for 
                    #the effect of batch size and the fact that the size of the last batch will not be equal to batch_size
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data) 
                    running_precision += stats_out['precision'] * inputs.size(0)
                    running_recall += stats_out['recall'] * inputs.size(0)
                    running_f1 += stats_out['f1-score'] * inputs.size(0)

                    bar.next()

            n = len(dataloaders_dict[phase].dataset)
            epoch_loss = float(running_loss / n)
            epoch_acc = float(running_corrects.double() / n)
            epoch_precision = (running_precision) / n         
            epoch_recall = (running_recall) / n        
            epoch_f1 = (running_f1) / n

        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_precision, epoch_recall, epoch_f1))
 
            # Save loss and acc to tensorboard log
            if phase == 'train':
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("Accuracy/train", epoch_acc, epoch)
                writer.add_scalar("Precision/train", epoch_precision , epoch)
                writer.add_scalar("Recall/train", epoch_recall, epoch)
                writer.add_scalar("F1/train", epoch_f1, epoch)
            else:
                writer.add_scalar("Loss/val", epoch_loss, epoch)
                writer.add_scalar("Accuracy/val", epoch_acc, epoch)
                writer.add_scalar("Precision/val", epoch_precision , epoch)
                writer.add_scalar("Recall/val", epoch_recall, epoch)
                writer.add_scalar("F1/val", epoch_f1, epoch)
              
            
            # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_precision > best_precision:
                best_precision = epoch_precision
                best_precision_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # Save only the model weights for easy loading into a new model
                final_out = {
                    'model': best_model_wts,
                    '__author__': 'Jamie R. Sykes',
                    '__model_name__': args.model_name                    
                    }       
                 
                PATH = os.path.join(model_path, args.model_name)
                with open(PATH + '.pkl', 'wb') as f:
                    pickle.dump(final_out, f)

                # Save the whole model with pytorch save function
                torch.save(model, PATH + '.pth')

            if phase == 'val':
                val_loss_history.append(epoch_loss)
    
        epoch += 1
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val recall_Acc: {:4f}'.format(best_precision_acc))
    print('Best val recall: {:4f}'.format(best_precision))
    
    # load best model weights and save
    model.load_state_dict(best_model_wts)
    
    writer.flush()
    writer.close()
    return model 

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.ToTensor(),
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=2) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda")

if args.arch == 'convnext_tiny':
    model_ft = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    in_feat = model_ft.classifier[2].in_features
    model_ft.classifier[2] = torch.nn.Linear(in_feat, num_classes)
elif args.arch == 'resnet18':
    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_feat = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_feat, num_classes)


#If checkpoint weights file exists, load these weights.
if args.cont_train == True and os.path.exists(os.path.join(model_path, args.model_name + '.pkl')) == True:
    pretrained_model_wts = pickle.load(open(os.path.join(model_path, args.model_name + '.pkl'), "rb"))
    unpickled_model_wts = copy.deepcopy(pretrained_model_wts['model'])

###############
#Remove 'module.' from layer names
    new_keys = []
    for key, value in unpickled_model_wts.items():
        new_keys.append(key.replace('module.', ''))
    for i in new_keys:
        unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)#    
##############
    model_ft.load_state_dict(unpickled_model_wts)
    print('Checkpint weights loaded')

#Run model on all GPUs
model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()

#Deine optimiser
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

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

criterion = nn.CrossEntropyLoss(weight=initial_bias)

# Train and evaluate
model = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, patience=args.patience, input_size=args.input_size, initial_bias=initial_bias)
