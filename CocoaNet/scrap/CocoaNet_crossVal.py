from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
from sklearn import metrics
import pandas as pd

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "/local/scratch/jrs596/dat/compiled_cocoa_images/CrossVal/"
data_dir '/local/scratch/jrs596/dat/compiled_cocoa_images/test_CrossVal/'

# File name for model
model_name = "CocoaNet50_CrossVal"

# Number of classes in the dataset
num_classes = 4

# Batch size for training (change depending on how much memory you have)
batch_size = 100

# Number of epochs to train for
min_epocs = 10
#Earley stopping
patience = 50 #epochs
beta = 1.005 ## % improvment in validation loss

input_size = 224
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

writer = SummaryWriter(log_dir='/local/scratch/jrs596/CocoaNet50_pretrained/logs_' + model_name)

results = pd.DataFrame(data=None, columns=['Acc', 'F1', 'Precision', 'Recall', 'Latency'])



def train_model(model, dataloaders, criterion, optimizer, patience, input_size, k):
    since = time.time()
    
    val_loss_history = []
    best_recall = 0.0
    best_acc = 0.0
    best_loss = 100.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_latencey = 0.0
    
    #Save Imagenet weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())

    ### Calculate and set bias for final layer based on imbalance in dataset classes
    dir_ = data_dir
    list_cats = []
    for i in sorted(os.listdir(dir_)):
        path, dirs, files = next(os.walk(dir_ + i))
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
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_precision = 0
            running_recall = 0
            running_f1 = 0
            running_latencey = 0
            # Iterate over data.
            
            for inputs, labels in dataloaders[phase]:
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
                    start = time.time()
                    outputs = model(inputs)
                    latencey = (inputs.size(0) / (time.time() - start)) * inputs.size(0) # Frames per second multplied by btach size

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #Calculate statistics
                #Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
                #running totals for these metrics by the total number of training or test samples. This controls for 
                #the effect of batch size and the fact that the size of the last batch will not be equal to batch_size
                
                running_latencey += latencey
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
                stats_out = stats['weighted avg']
                running_precision += stats_out['precision'] * inputs.size(0)
                running_recall += stats_out['recall'] * inputs.size(0)
                running_f1 += stats_out['f1-score'] * inputs.size(0)

            n = len(dataloaders_dict[phase].dataset)
            epoch_loss = float(running_loss / n)
            epoch_acc = float(running_corrects.double() / n)
            epoch_precision = (running_precision) / n         
            epoch_recall = (running_recall) / n        
            epoch_f1 = (running_f1) / n
            epoch_latencey = running_latencey / n
            
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_precision, epoch_recall, epoch_f1))
 
            # Save loss and acc to tensorboard log
            k = str(k)
            if phase == 'train':
                writer.add_scalar(k + "_Loss/train", epoch_loss, epoch)
                writer.add_scalar(k + "_Accuracy/train", epoch_acc, epoch)
                writer.add_scalar(k + "_Precision/train", epoch_precision , epoch)
                writer.add_scalar(k + "_Recall/train", epoch_recall, epoch)
                writer.add_scalar(k + "_F1/train", epoch_f1, epoch)
            else:
                writer.add_scalar(k + "_Loss/val", epoch_loss, epoch)
                writer.add_scalar(k + "_Accuracy/val", epoch_acc, epoch)
                writer.add_scalar(k + "_Precision/val", epoch_precision , epoch)
                writer.add_scalar(k + "_Recall/val", epoch_recall, epoch)
                writer.add_scalar(k + "_F1/val", epoch_f1, epoch)
              
            
            # Update best weights only if recall has improved
            if phase == 'val' and epoch_recall > best_recall:
                best_recall = epoch_recall
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_precision = epoch_precision
                best_recall = epoch_recall
                best_f1 = epoch_f1
                best_latencey = epoch_latencey

                best_model_wts = copy.deepcopy(model.state_dict())

                val_loss_history.append(epoch_loss)

        epoch += 1
        
    output = [best_acc, best_f1, best_precision, best_recall, best_latencey]
    writer.flush()
    writer.close()
    return output



#Set Model Parameters’ .requires_grad attribute
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


#Initialize and Reshape the Networks
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract) # Not requiered for full fine tuning
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft #, input_size

# Initialize the model for this run
model_ft = initialize_model(num_classes, feature_extract, use_pretrained=True)


print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
# Create training and validation dataloaders
def load_data(data_dir):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }   

    print("Initializing Datasets and Dataloaders...")   

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
        data_transforms[x]) for x in ['train', 'val']}  

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
        batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    return dataloaders_dict

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)



#Run Training and Validation Step
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

temp_dir = '/local/scratch/jrs596/dat/compiled_cocoa_images/CrossVal_epoch_randomisation/'

def resample_data(temp_dir, k, split):
    try:
        shutil.rmtree(temp_dir + 'train')
        shutil.rmtree(temp_dir + 'val')
    except:
        pass

    for i in ['FPR', 'WBD', 'Healthy', 'BPR']:
        image_list = os.listdir(data_dir + '/' + i)
        #random.shuffle(image_list)
        for j in ['train', 'val']:
            os.makedirs(temp_dir + j + '/' + i, exist_ok = True)
            if j == 'val':

                sample = image_list[:int(len(image_list)*split)]  
                if split > 0.1:
                    sample = sample[int(len(image_list)*(split-0.1)):]
                for s in sample:
                    source = data_dir + '/' + i + '/' + s
                    dest = temp_dir + j + '/' + i + '/' + s
                    shutil.copy(source, dest)

                for im in image_list:
                    if im not in sample:
                        source = data_dir + '/' + i + '/' + im
                        dest = temp_dir + 'train/' + i + '/' + im
                        shutil.copy(source, dest)
    
    dataloaders = load_data(temp_dir)
    return dataloaders


# Train and evaluate


split = 0
for k in range(10):
    split += 0.1
    dataloaders_dict = randomise_data(temp_dir, k, split)

    out = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, patience=patience, input_size=input_size, k)
    results.loc[len(results)] = out
    results.to_csv('/local/scratch/jrs596/CrossVal/' + model_name + '_results.csv')


