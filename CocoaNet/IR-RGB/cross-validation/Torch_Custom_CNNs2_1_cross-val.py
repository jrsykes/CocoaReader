
from __future__ import print_function
from __future__ import division

import os
import yaml
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, ResNet50_Weights
import time
import copy
import wandb
from sklearn import metrics
from progress.bar import Bar
#from torchvision.models.convnext import _convnext, CNBlockConfig
import argparse
import pickle
import json
import sys
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
from ColorGradingLayer import CGResNet18
from collections import Counter

parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='test',
                        help='save name for model')
parser.add_argument('--project_name', type=str, default=None,
                        help='Name for wandb project')
parser.add_argument('--run_name', type=str, default=None,
                        help='Name for wandb run')
parser.add_argument('--sweep', action='store_true', default=False,
                        help='Run Waits and Biases optimisation sweep')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--sweep_config', type=str, default=None,
                        help='.yml sweep configuration file')
parser.add_argument('--sweep_count', type=int, default=100,
                        help='Number of models to train in sweep')
parser.add_argument('--root', type=str, default='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat',
                        help='location of all data')
parser.add_argument('--data_dir', type=str, default='test',
                        help='location of all data')
parser.add_argument('--custom_pretrained', action='store_true', default=False,
                        help='Train useing specified pre-trained weights?')
parser.add_argument('--custom_pretrained_weights', type=str,
                        help='location of pre-trained weights')
parser.add_argument('--quantise', action='store_true', default=False,
                        help='Train with Quantization Aware Training?')
parser.add_argument('--batch_size', type=int, default=4,
                        help='Initial batch size')
parser.add_argument('--max_epochs', type=int, default=500,
                        help='n epochs before early stopping')
parser.add_argument('--min_epochs', type=int, default=10,
                        help='n epochs before loss is assesed for early stopping')
parser.add_argument('--patience', type=int, default=1,
                        help='n epochs to run without improvment in loss')
parser.add_argument('--beta', type=float, default=1.00,
                        help='minimum required per cent improvment in validation loss')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps, Default:1e-8')
parser.add_argument('--batchnorm_momentum', type=float, default=1e-1,
                        help='Batch norm momentum hyperparameter for resnets, Default:1e-1')
parser.add_argument('--input_size', type=int, default=277,
                        help='image input size')
parser.add_argument('--delat', type=float, default=1.4,
                        help='delat for dynamic focal loss')
parser.add_argument('--arch', type=str, default='convnext_tiny',
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


args = parser.parse_args()
print(args)

def setup(args):
    data_dir = os.path.join(args.root, args.data_dir)
    #Define some variable and paths
    os.environ['TORCH_HOME'] = os.path.join(args.root, "TORCH_HOME")

    num_classes= len(os.listdir(data_dir + '/fold_0/train'))

    # Specify whether to use GPU or cpu. Quantisation aware training is not yet avalable for GPU.
   
    device = torch.device("cuda")
    
    ### Calculate and set bias for final layer based on imbalance in dataset classes
    dir_ = os.path.join(data_dir, 'fold_0' ,'train')
    list_cats = []
    for i in sorted(os.listdir(dir_)):
        _, _, files = next(os.walk(os.path.join(dir_, i)))
        list_cats.append(len(files))

    weights = []
    for i in list_cats:
        weights.append(np.log((max(list_cats)/i)))

    initial_bias = torch.FloatTensor(weights).to(device)

    return data_dir, num_classes, initial_bias, device


def train_model(model, optimizer, device, dataloaders_dict, criterion, patience, initial_bias, input_size, batch_size, n_tokens=None, AttNet=None, ANoptimizer=None, num_classes=None):   
    val_loss_history = []
    best_f1 = 0.0
    best_f1_acc = 0.0
    best_train_metrics = {'loss': 0.0, 'acc': 0.0, 'f1': 0.0, 'recall': 0.0, 'precision': 0.0, 'BPR_F1': 0, 'FPR_F1': 0, 'Healthy_F1': 0, 'WBD_F1': 0}
    best_val_metrics = {'loss': 0.0, 'acc': 0.0, 'f1': 0.0, 'recall': 0.0, 'precision': 0.0, 'BPR_F1': 0, 'FPR_F1': 0, 'Healthy_F1': 0, 'WBD_F1': 0}

    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts['module.fc.bias'] = initial_bias.to(device)

    class_counts = class_count(dataloaders_dict)
    
    epoch = 0
    #while patience >= 0: # Run untill validation loss has not improved for n epochs equal to patience variable and batchsize has decaed to 1
    while patience > 0 and epoch < args.max_epochs:
        print('\nEpoch {}'.format(epoch))
        print('-' * 10) 
        step  = 0
        class_f1_scores = {'0': 0, '1': 0, '2': 0, '3': 0}

        #Ensure minimum number of epochs is met before patience is allow to reduce
        if len(val_loss_history) > args.min_epochs:
           #If the current loss is not at least 0.5% less than the lowest loss recorded, reduce patiece by one epoch
            if val_loss_history[-1] > min(val_loss_history)*args.beta:
                patience -= 1
            elif val_loss_history[-1] == np.nan:
                patience -= 1
            else:
               #If validation loss improves by at least 0.5%, reset patient to initial value
               patience = args.patience
        print('Patience: ' + str(patience) + '/' + str(args.patience))
       #If patience gets to 6, half batch size, reintialise dataloader and revert to best model weights to undo any overfitting.
       #Do this for every epoch where patience is less than 6. i.e. if inital batch size = 128, the last epoch will have a batchsize of at most 2.

       #Training and validation loop
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                if AttNet != None:
                    AttNet.train()
 
            elif phase == 'val':
               #Model quatisation with quantisation aware training
                model.eval()   # Set model to evaluate mode
                if AttNet != None:
                    AttNet.eval()

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
                    #split images in bacth into 16 non-overlapping chunks then recombine into bacth
                    if args.split_image == True:
                        
                        #split batch into list of tensors
                        inputs = torch.split(inputs, 1, dim=0)
                        #print(inputs[0].shape)
                        token_size = int(input_size / n_tokens)
                        x, y, h ,w = 0, 0, token_size, token_size
                        ims = []
                        for t in inputs:
                            #crop im to 16 non-overlapping 277x277 tensors
                            for i in range(n_tokens):
                                for j in range(n_tokens):
                                    t.squeeze_(0)
                                    im1 = t[:, x:x+h, y:y+w]
                                    ims.append(im1)
                                    y += w
                                y = 0
                                x += h
                            x, y, h ,w = 0, 0, token_size, token_size
                        #convert list of tensors into a batch tensor of shape (512, 3, 277, 277)
                        inputs = torch.stack(ims, dim=0)
                    
                    #Load images and lables from current batch onto GPU(s)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()       
                    if ANoptimizer != None:
                        ANoptimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                       # Get model outputs and calculate loss
                       # In train mode we calculate the loss by summing the final output and the auxiliary output
                       # but in testing we only consider the final output.
                        outputs = model(inputs)

                        #Reduce dimensionality of output to 32x2
                        if args.split_image == True:
                            #old_output_shell = outputs[0:batch_size,0:num_classes]
                            new_batch = []
                            for i in range(batch_size):
                                outputs_ = outputs[i*n_tokens**2:(i+1)*n_tokens**2]
                                outputs_ = outputs_.detach().cpu()
                                
                                outputs_flat = torch.empty(0)
                                for i in range(outputs_.shape[1]):
                                    outputs_flat = torch.cat((outputs_flat,outputs_[:,i]),0)

                                outputs_ = outputs_flat.to(device).unsqueeze(0)
                                outputs_ = AttNet(outputs_)
                                                        
                                new_batch.append(outputs_)

                            outputs = torch.stack(new_batch, dim=0).squeeze(1)

                        loss = criterion(outputs, labels)
          
                        _, preds = torch.max(outputs, 1)    
                        stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)                      
                        stats_out = stats['weighted avg']                 

                       # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()  
                            if ANoptimizer != None:
                                ANoptimizer.step() 
      

                   #Calculate statistics
                   #Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
                   #running totals for these metrics by the total number of training or validation samples. This controls for 
                   #the effect of batch size and the fact that the size of the last batch will less than args.batch_size
                    current_batch_size = len(labels)
                    running_loss += loss.item() * current_batch_size 
                    running_corrects += torch.sum(preds == labels.data) 
                    running_precision += stats_out['precision'] * current_batch_size 
                    running_recall += stats_out['recall'] * current_batch_size 
                    running_f1 += stats_out['f1-score'] * current_batch_size 
                    
                    #get per class f1 scores from stats dictionary
                    for i in range(num_classes):
                        try: 
                            class_f1_scores[str(i)] += stats[str(i)]['f1-score'] * stats[str(i)]['support']
                        except:
                            pass

                    
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
            
            #Calculate per class f1 scores
            for key in class_f1_scores.keys():
               class_f1_scores[key] = class_f1_scores[key] / class_counts[phase][int(key)]
            
            # Save statistics to wandb log
            if phase == 'train':
                current_train_metrics = {'epoch': epoch, 'loss': epoch_loss, 'acc': epoch_acc, 'precision': epoch_precision, 'recall': epoch_recall, 'f1': epoch_f1,
                                         'BPR_F1': class_f1_scores['0'], 'FPR_F1': class_f1_scores['1'], 'Healthy_F1': class_f1_scores['2'], 'WBD_F1': class_f1_scores['3']}

           # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_f1_acc = epoch_acc
                #define best train metrics dictionary
                best_train_metrics = current_train_metrics
                #define best val metrics dictionary
                best_val_metrics = {'epoch': epoch, 'loss': epoch_loss, 'acc': epoch_acc, 'precision': epoch_precision, 'recall': epoch_recall, 'f1': epoch_f1,
                                    'BPR_F1': class_f1_scores['0'], 'FPR_F1': class_f1_scores['1'], 'Healthy_F1': class_f1_scores['2'], 'WBD_F1': class_f1_scores['3']}
            
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            
            if phase == 'train':
                wandb.log({"epoch": epoch, "Train_loss": epoch_loss, "Train_acc": epoch_acc, "Train_F1": epoch_f1})  
            else:
                wandb.log({"epoch": epoch, "Val_loss": epoch_loss, "Val_acc": epoch_acc, "Val_F1": epoch_f1, "Best_F1": best_f1, "Best_F1_acc": best_f1_acc})


        bar.finish() 
        epoch += 1

    PATH = os.path.join('/local/scratch/jrs596/dat/IR_RGB_Comp_data/cross-val_models', wandb.run.name)    
    torch.save(model.module, PATH + '.pth') 
    
    return best_train_metrics, best_val_metrics

def class_count(dataloaders_dict):
    train_class_counts = Counter()
    val_calss_counts = Counter()

    for _, labels in dataloaders_dict['train']:
        train_class_counts.update(Counter(labels.numpy()))
    
    for _, labels in dataloaders_dict['val']:
        val_calss_counts.update(Counter(labels.numpy()))

    class_counts = {'train': train_class_counts, 'val': val_calss_counts}
    return class_counts




def build_datasets(input_size, data_dir):
    # Data augmentation and normalization for training
    # Just normalization for device
    
    # matrix = torch.tensor([[1.20626902580261, 0.69084495306015, -0.367000162601471], 
    #           [0.864837288856506, 0.967565953731537, 0.168635278940201], 
    #           [-0.376707583665848, 0.264213144779205,-1.11190068721771]])
    
    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize((input_size,input_size)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            #CrossTalkColorGrading(matrix)

        ]),
        'val': transforms.Compose([
            #transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
            #CrossTalkColorGrading(matrix)
        ]),
    }   

    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    return image_datasets


    

def Remove_module_from_layers(unpickled_model_wts):
    new_keys = []
    for key, value in unpickled_model_wts.items():
        new_keys.append(key.replace('module.', ''))
    for i in new_keys:
        unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)
    return unpickled_model_wts

def build_model(num_classes, device):
    
    if args.arch == 'convnext_tiny':
        print('Loaded ConvNext Tiny with pretrained IN weights')
        model_ft = models.convnext_tiny(weights = None)
        in_feat = model_ft.classifier[2].in_features
        model_ft.classifier[2] = torch.nn.Linear(in_feat, num_classes)
    elif args.arch == 'resnet18':
        print('Loaded ResNet18 with pretrained IN weights')
        model_ft = models.resnet18(weights=None)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif args.arch == 'resnet50':
        print('Loaded ResNet50 with pretrained IN weights')
        model_ft = models.resnet50(weights=None)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif args.arch == 'CGresnet18':
        model_ft = CGResNet18(num_classes=num_classes)
    else:
        print("Architecture name not recognised")
        exit(0)
    # Load custom pretrained weights    

    model_ft = nn.DataParallel(model_ft)
    model_ft = set_batchnorm_momentum(model_ft, momentum=args.batchnorm_momentum)
    model_ft = model_ft.to(device)
    return model_ft

def set_batchnorm_momentum(self, momentum):
    for m in self.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = momentum
    return self

class AttentionNet(nn.Module):
    def __init__(self, num_classes, num_tokens):
        super(AttentionNet, self).__init__()
        num_heads = num_classes
        embed_dim = num_tokens**2*num_classes
        head_dim = embed_dim//num_heads
        # define linear transformations for queries, keys, and values
        self.query_transform = nn.Linear(embed_dim, num_heads * head_dim)
        self.key_transform = nn.Linear(embed_dim, num_heads * head_dim)
        self.value_transform = nn.Linear(embed_dim, num_heads * head_dim)
        #apply muti-AttentionNet head
        self.Attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        #apply layer normalisation
        self.layernorm = nn.LayerNorm(embed_dim)
        #apply linear layer
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        #apply AttentionNet
        queries = self.query_transform(x)
        keys = self.key_transform(x)
        values = self.value_transform(x)
        x, _ = self.Attention(queries, keys, values)
        x = self.fc1(x)
        #apply layer normalisation
        x = self.layernorm(x)
        #apply linear layer
        x = self.fc2(x)
        #apply relu activation
        x = F.softmax(x, dim=1)
        return x


class DynamicFocalLoss(nn.Module):
    def __init__(self, delta=1, dataloader=None):
        super(DynamicFocalLoss, self).__init__()
        self.delta = delta
        self.dataloader = dataloader
        self.weights_dict = {}

    def forward(self, inputs, targets, step):
        loss = nn.CrossEntropyLoss()(inputs, targets)
        # Update weights_dict based on targets and predictions
        preds = torch.argmax(inputs, dim=1)
        for i in range(inputs.size(0)):
            #get filename from dataset
            filename = self.dataloader.dataset.samples[step + i][0].split("/")[-1]
            if filename not in self.weights_dict:
                self.weights_dict[filename] = 1
            if preds[i] != targets[i]:
                self.weights_dict[filename] += self.delta
        step += inputs.size(0)
        
        # Apply weights to loss based on weights_dict
        weighted_loss = torch.zeros(1).to(loss.device)
        for filename, weight in self.weights_dict.items():
            if weight > 1:
                weighted_loss += loss * weight
            else:
                weighted_loss += loss
        weighted_loss /= len(self.weights_dict)

        return weighted_loss, step


def train():
    data_dir, num_classes, initial_bias, device = setup(args)
    criterion = nn.CrossEntropyLoss()

    # Initialize lists to store results
    train_metrics_dict = {'loss': [], 'f1': [], 'acc': [], 'precision': [], 'recall': [], 'BPR_F1': [], 'FPR_F1': [], 'Healthy_F1': [], 'WBD_F1': []}
    val_metrics_dict = {'loss': [], 'f1': [], 'acc': [], 'precision': [], 'recall': [], 'BPR_F1': [], 'FPR_F1': [], 'Healthy_F1': [], 'WBD_F1': []}
        
    for fold in range(10):
        print(f'Fold {fold}')
        model = build_model(num_classes, device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                                weight_decay=args.weight_decay, eps=args.eps)

        
        wandb.init(project=args.project_name, name=args.run_name+f'_fold_{fold}')
   
        # Create training and validation datasets using the current fold
        image_datasets = build_datasets(input_size=args.input_size, data_dir=os.path.join(data_dir, f'fold_{fold}'))
    
        # Create dataloaders for the training and validation datasets
        dataloaders_dict = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=False),
            'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=False)
        }

        # Train the model and store the results
        best_train_metrics, best_val_metrics = train_model(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloaders_dict=dataloaders_dict,
            criterion=criterion,
            patience=args.patience,
            initial_bias=initial_bias,
            input_size=None,
            n_tokens=None,
            batch_size=args.batch_size,
            AttNet=None,
            ANoptimizer=None,
            num_classes=num_classes
        )

        wandb.finish()

        # Store the results for this fold
        for metric in train_metrics_dict:
            train_metrics_dict[metric].append(best_train_metrics[metric])
        for metric in val_metrics_dict:
            val_metrics_dict[metric].append(best_val_metrics[metric])                                                   
                                                       
    #Divide all values in the dictionaries by 10 to get the mean
    mean_train_metrics_dict = {}
    mean_val_metrics_dict = {}

    for metric in train_metrics_dict:
        mean_train_metrics_dict[metric] = np.mean(train_metrics_dict[metric])
    for metric in val_metrics_dict:
        mean_val_metrics_dict[metric] = np.mean(val_metrics_dict[metric])

    #Calculate standard error metrics dict
    train_se_metrics_dict = {}
    val_se_metrics_dict = {}

    for metric in train_metrics_dict:
        train_se_metrics_dict[metric] = np.std(train_metrics_dict[metric]) / np.sqrt(len(train_metrics_dict[metric]))
    for metric in val_metrics_dict:
        val_se_metrics_dict[metric] = np.std(val_metrics_dict[metric]) / np.sqrt(len(val_metrics_dict[metric]))


    print()
    print(f'Mean train metrics: {mean_train_metrics_dict}')
    print(f'Standard error train metrics: {train_se_metrics_dict}')
    print()
    print(f'Mean val metrics: {mean_val_metrics_dict}')
    print(f'Standard error val metrics: {val_se_metrics_dict}')
    print()
          
    run = wandb.init(project=args.project_name)
    artifact = wandb.Artifact(args.run_name + '_results', type='dataset')

    # Log the results as wandb artifacts
    mean_dict = {'train_mean_metrics': mean_train_metrics_dict, 'val_mean_metrics': mean_val_metrics_dict,
                     'train_se_metrics': train_se_metrics_dict, 'val_se_metrics': val_se_metrics_dict}
    
    with open(args.run_name + '_results_dict.json', 'w') as f:
        json.dump(mean_dict, f)

    artifact.add_file(args.run_name + '_results_dict.json')
    run.log_artifact(artifact)

    wandb.finish()
    os.remove(args.run_name + '_results_dict.json')

train()

