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

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "/local/scratch/jrs596/dat/compiled_cocoa_images/split" #even_split"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 4

# Batch size for training (change depending on how much memory you have)
batch_size = 42

# Number of epochs to train for
num_epochs = 1000
min_epocs = 100
patience = 25 #epochs
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

writer = SummaryWriter(log_dir='/local/scratch/jrs596/CocoaNet50_pretrained/logs')
#writer = SummaryWriter(log_dir='/local/scratch/jrs596/CocoaNet50_ImageNet/logs')


def train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, is_inception=False):
    since = time.time()
    
    train_loss_history = [100]
    loss_to_beat = 100
    train_loss = 0
    best_loss = 100
    best_recall = 0.0
    best_recall_acc = 0.0

    #best_model_wts = copy.deepcopy(model.state_dict()) #### Re-loads ImageNet weights

 
    # load pretrained weights from ResDesNet50 
    pretrained_model_path = '/local/scratch/jrs596/ResNetFung50_Torch/models/model.pkl'
    pretrained_model_wts = pickle.load(open(pretrained_model_path, "rb"))
    best_model_wts = copy.deepcopy(pretrained_model_wts['model'])



    ### Calculate and set bias for final layer
    dir_ = data_dir + '/train/'
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



    epoch = num_epochs
    patience_ = patience
    while epoch > 0 and patience_ > 0: # train_loss < loss_to_beat * 0.8:
        print('Epoch {}/{}'.format(num_epochs-epoch+1, num_epochs))
        print('-' * 10)

        if len(train_loss_history) > min_epocs:
            if train_loss_history[-1] >= train_loss_history[-2]*1.01:
                patience_ -= 1
            else:
                patience_ = patience
        print('Patience: ' + str(patience_) + '/' + str(patience))

#        if len(train_loss_history) > patience + 5:
#            loss_to_beat = train_loss_history[-patience]
#        else:
#            loss_to_beat = train_loss_history[0]#

#        if len(train_loss_history) > patience + 5: 
#            print('\nLoss to beat: ' + str(round(loss_to_beat* 0.8, 2)) + '\n')

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
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
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

        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_precision, epoch_recall, epoch_f1))
 
            # Save loss and acc to tensorboard log
            if phase == 'train':
                writer.add_scalar("Loss/train", epoch_loss, num_epochs-epoch+1)
                writer.add_scalar("Accuracy/train", epoch_acc, num_epochs-epoch+1)
                writer.add_scalar("Precision/train", epoch_precision , num_epochs-epoch+1)
                writer.add_scalar("Recall/train", epoch_recall, num_epochs-epoch+1)
                writer.add_scalar("F1/train", epoch_f1, num_epochs-epoch+1)
            else:
                writer.add_scalar("Loss/val", epoch_loss, num_epochs-epoch+1)
                writer.add_scalar("Accuracy/val", epoch_acc, num_epochs-epoch+1)
                writer.add_scalar("Precision/val", epoch_precision , num_epochs-epoch+1)
                writer.add_scalar("Recall/val", epoch_recall, num_epochs-epoch+1)
                writer.add_scalar("F1/val", epoch_f1, num_epochs-epoch+1)
              
            
            # Save model only if accuracy has improved
            #if phase == 'val' and epoch_acc > best_acc:
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                final_out = {
                    'model': best_model_wts,
                    '__author__': 'Jamie R. Sykes'                    
                    }

                model_path = '/local/scratch/jrs596/CocoaNet50_pretrained/models/CocoaNet50_pretrained_model.pkl'
                #model_path = '/local/scratch/jrs596/CocoaNet50_ImageNet/models/CocoaNet50_ImageNet_model.pkl'

                with open(model_path, 'wb') as f:
                    pickle.dump(final_out, f)

                
                torch.save(model, '/local/scratch/jrs596/CocoaNet50_pretrained/models/CocoaNet50_pretrained_model.pth')
                #torch.save(model, '/local/scratch/jrs596/CocoaNet50_ImageNet/models/CocoaNet50_ImageNet_model.pth')
       
            if phase == 'val':
                train_loss_history.append(epoch_loss)
                train_loss = epoch_loss

        epoch -= 1
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val recall_Acc: {:4f}'.format(best_recall_acc))
    #print('Best val recall: {:4f}'.format(best_recall))
    
    # load best model weights and save
    model.load_state_dict(best_model_wts)
    
    writer.flush()
    writer.close()
    return model, train_loss_history



#Set Model Parameters’ .requires_grad attribute
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


#Initialize and Reshape the Networks
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract) # Not requiered for full fine tuning
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
#print(model_ft)

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
#print("Params to learn:")
#if feature_extract:
#    params_to_update = []
#    for name,param in model_ft.named_parameters():
#        if param.requires_grad == True:
#            params_to_update.append(param)
#            print("\t",name)
#else:
#    for name,param in model_ft.named_parameters():
#        if param.requires_grad == True:
#            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)



#Run Training and Validation Step
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate


model, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False)


