from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
from sklearn import metrics
from progress.bar import Bar
#from torchvision.models import ConvNeXt_Tiny_Weights

from torchvision.models import ResNet18_Weights

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure


# File name for model
model_name = "PlantNotPlant3.2"

root = "/scratch/staff/jrs596/dat"
data_dir = os.path.join(root, model_name, 'split')
model_path = os.path.join(root, 'models')
log_dir= os.path.join(model_path, "logs", "logs_" + model_name)

# Number of classes in the dataset
num_classes = len(os.listdir(os.path.join(data_dir, 'val')))

# Batch size for training (change depending on how much memory you have)
batch_size = 42

# Number of epochs to train for
min_epocs = 10
#Earley stopping
patience = 20 #epochs
beta = 1.005 ## % improvment in validation loss

input_size = 224
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
#feature_extract = False

writer = SummaryWriter(log_dir=log_dir)

def train_model(model, dataloaders, criterion, optimizer, patience, input_size):
    since = time.time()
    
    val_loss_history = []
    best_precision = 0.0
    best_model_acc = 0.0
    
    #Save Imagenet weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())

    ### Calculate and set bias for final layer based on imbalance in dataset classes
    dir_ = os.path.join(data_dir, 'train')
    list_cats = []
    for i in sorted(os.listdir(dir_)):
        path, dirs, files = next(os.walk(os.path.join(dir_, i)))
        list_cats.append(len(files))
    
    weights = []
    for i in list_cats:
        weights.append(np.log((max(list_cats)/i)))

    initial_bias = torch.FloatTensor(weights)
    best_model_wts['module.fc.bias'] = initial_bias
    #######################################################################
    
    epoch = 0
    initial_patience = patience
    while patience > 0: # Run untill validation loss has not improved for n epochs equal to patience variable
        print('\nEpoch {}'.format(epoch))
        print('-' * 10)

        if len(val_loss_history) > min_epocs:
            if val_loss_history[-1] > min(val_loss_history)*beta:
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
            with Bar('Learning...', max=n/batch_size) as bar:
                print()
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
                        loss += (1-stats_out['precision'])*0.4

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
                best_model_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                print('Saving')        
                # Save only the model weights for easy loading into a new model
                final_out = {
                    'model': best_model_wts,
                    '__author__': 'Jamie R. Sykes',
                    '__model_name__': model_name                    
                    }    
            
                with open(os.path.join(model_path, model_name + '.pkl'), 'wb') as f:
                    pickle.dump(final_out, f)

                # Save the whole model with pytorch save function
                torch.save(model, os.path.join(model_path, model_name + '.pth'))

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                
    
        epoch += 1
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best model Acc: {:4f}'.format(best_model_acc))
    print('Best val Precision: {:4f}'.format(best_precision))
    
    # load best model weights and save
    model.load_state_dict(best_model_wts)
    


    writer.flush()
    writer.close()
    return model #, val_loss_history



#Set Model Parameters’ .requires_grad attribute
#def set_parameter_requires_grad(model, feature_extracting):
#    if feature_extracting:
#        for param in model.parameters():
#            param.requires_grad = False


#Initialize and Reshape the Networks
#def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

model_ft = None
model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
#set_parameter_requires_grad(model_ft, feature_extract) # Not requiered for full fine tuning
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
#    #model_ft.fc[1] = torch.nn.Sigmoid(1)
#    return model_ft #, input_size

# Initialize the model for this run

#model_ft = initialize_model(num_classes, feature_extract, use_pretrained=True)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda")

#Run model on all GPUs
model_ft = nn.DataParallel(model_ft)

#Create the Optimizer
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
#print("Params to learn:")
#if feature_extract:
#    params_to_update = []
#    for name,param in model_ft.named_parameters():
#        if param.requires_grad == True:
#            params_to_update.append(param)
#            print("\t",name)
#else:
 #   for name,param in model_ft.named_parameters():
        #if param.requires_grad == True:
            #print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)



#Run Training and Validation Step
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate


model = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, patience=patience, input_size=input_size)
