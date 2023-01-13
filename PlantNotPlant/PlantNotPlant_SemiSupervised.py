from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
import time
import os
import copy
import shutil
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
from sklearn import metrics
from progress.bar import Bar


# File name for model
model_name = "PlantNotPlant_SemiSup"
pretrained_model_path = '/local/scratch/jrs596/dat/models/PlantNotPlant_SemiSup.pkl'

pretrained_model_wts = pickle.load(open(pretrained_model_path, "rb"))
unpickled_model_wts = copy.deepcopy(pretrained_model_wts['model'])

###############
#Remove 'module.' from layer names
new_keys = []
for key, value in unpickled_model_wts.items():
    new_keys.append(key.replace('module.', ''))
for i in new_keys:
    unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)#    

##############


root = "/local/scratch/jrs596/dat"
data_dir = os.path.join(root, 'PlantNotPlant3.3', 'train_full')
GoogleBing_data_dir = os.path.join(root, 'Forestry_ArableImages_GoogleBing_clean/train')


model_path = os.path.join(root, 'models')
log_dir= os.path.join(model_path, "logs", "logs_" + model_name)

# Number of classes in the dataset
num_classes = len(os.listdir(data_dir))#, 'train_full')))


# Batch size for training (change depending on how much memory you have)
batch_size = 42

# Number of epochs to train for
min_epocs = 1
#Earley stopping
patience = 6 #epochs
beta = 1.005 ## % improvment in validation loss

input_size = 224


writer = SummaryWriter(log_dir=log_dir)

def train_model(model, dataloaders, criterion, optimizer, patience, input_size):
    since = time.time()
    
    train_loss_history = []
    best_recall = 0.0
    best_model_acc = 0.0
    epoch = 0
    initial_patience = patience
    
    while patience > 0: # Run untill validation loss has not improved for n epochs equal to patience variable
        print('\nMinor epoch {}'.format(epoch))
        print('-' * 10)

        if len(train_loss_history) > min_epocs:
            if train_loss_history[-1] > min(train_loss_history)*beta:
                patience -= 1
            else:
                patience = initial_patience
        print('Patience: ' + str(patience) + '/' + str(initial_patience))
        
        # Each epoch has a training and validation phase
        
        for phase in ['train']:
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
                        loss += (1-stats_out['recall'])*0.4

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
            if phase == 'train' and epoch_recall > best_recall:
                best_recall = epoch_recall
                best_model_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                #PATH = '/local/scratch/jrs596/ResNetFung50_Torch/models/'

                # Save only the model weights for easy loading into a new model
                final_out = {
                    'model': best_model_wts,
                    '__author__': 'Jamie R. Sykes',
                    'model_name': model_name

                    }    
                 
                with open(os.path.join(model_path, model_name + '.pkl'), 'wb') as f:
                    pickle.dump(final_out, f)

                # Save the whole model with pytorch save function
                #torch.save(model, os.path.join(model_path, model_name + '.pth'))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                
    
        epoch += 1
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best model Acc: {:4f}'.format(best_model_acc))
    print('Best val Recall: {:4f}'.format(best_recall))
    
    # load best model weights and save
    model.load_state_dict(best_model_wts)

    writer.flush()
    writer.close()
    return model


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ]),
}

print("Initializing Datasets and Dataloaders...")



#Define and load model
model_ft = models.resnet18(weights=None)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft.load_state_dict(unpickled_model_wts)

device = torch.device("cuda")
#Run model on all GPUs
model_ft = nn.DataParallel(model_ft)
#Create the Optimizer
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


with open(os.path.join(root, 'im_move_log.csv'), 'w') as f:
    f.write('Plant,NotPlant\n')

def relable(model, e):
    dataset = datasets.ImageFolder(GoogleBing_data_dir, transform=data_transforms['val'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    plant, notplant = 0, 0
    print('Relableing')
    for i, (inputs, labels) in enumerate(loader):
        source, _ = loader.dataset.samples[i]
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        # [0][1] = Plant
        # [0][0] = NotPlant
        if outputs[0][1].item() > 0.99:
            dest = os.path.join(data_dir, 'Plant', str(time.time()) + '.jpg')
            test_dest = os.path.join(root, 'PNP_SemiSupervised_moved/Plant', e)
            os.makedirs(test_dest, exist_ok=True)
            shutil.copy(source, os.path.join(test_dest, str(time.time()) + '.jpg'))
            shutil.move(source, dest)
            plant += 1
        if outputs[0][0].item() > 0.99:
            dest = os.path.join(data_dir, 'NotPlant', str(time.time()) + '.jpg')
            test_dest = os.path.join(root, 'PNP_SemiSupervised_moved/NotPlant', e)
            os.makedirs(test_dest, exist_ok=True)
            shutil.copy(source, os.path.join(test_dest, str(time.time()) + '.jpg'))
            shutil.move(source,dest)
            notplant += 1
    print('Moved {} plant and {} none plant images'.format(plant, notplant))    
    with open('im_move_log.csv', 'a') as f:
        f.write('{},{}\n'.format(plant, notplant))
    return plant+notplant


moved = 1
e = 15
while moved > 0:    
    print('\nMajor epoch {}'.format(e))
    e += 1
    
    if moved == 0:
        break
    #Reload dataset after relabeling
    image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['train'])
    dataloaders_dict = {'train': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

    # Train and evaluate
    model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, patience=patience, input_size=input_size)

    moved = relable(model_ft, str(e))