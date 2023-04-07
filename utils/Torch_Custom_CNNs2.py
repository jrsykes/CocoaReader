from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torchvision.models import ResNet18_Weights, ResNet50_Weights ,ConvNeXt_Tiny_Weights
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse
import yaml
from sklearn.decomposition import PCA
import torchvision.ops as ops


parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='test',
                        help='save name for model')
parser.add_argument('--project_name', type=str, default=None,
                        help='Name for wandb project')
parser.add_argument('--sweep', action='store_true', default=False,
                        help='Run Waits and Biases optimisation sweep')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--sweep_config', type=str, default='/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/token_sweep_config.yml',
                        help='.yml sweep configuration file')
parser.add_argument('--sweep_count', type=int, default=100,
                        help='Number of models to train in sweep')
parser.add_argument('--root', type=str, default='/local/scratch/jrs596/dat',
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
parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps, Default:1e-8')
parser.add_argument('--input_size', type=int, default=64,
                        help='image input size')
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


args = parser.parse_args()
print(args)

def setup(args):
    data_dir = os.path.join(args.root, args.data_dir)
    #Set environment variables for wandb sweep
    os.environ['WANDB_CACHE_DIR'] = os.path.join(args.root, 'WANDB_CACHE')
    os.environ['WANDB_DIR'] = os.path.join(args.root, 'WANDB_DIR')
    #Define some variable and paths
    os.environ['TORCH_HOME'] = os.path.join(args.root, "TORCH_HOME")
    
    data_dir = os.path.join(args.root, args.data_dir)
    num_classes= len(os.listdir(data_dir + '/train'))

    
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

    
    #criterion = nn.CrossEntropyLoss()
    criterion = DynamicFocalLoss(alpha=wandb.config.alpha, gamma=wandb.config.gamma)

    # define your hyperparameters
    hyperparams = {
        'learning_rate': 1e-5,
        'weight_decay': 0.0001,
        'eps': 1e-8
        }

    return data_dir, num_classes, initial_bias, criterion, hyperparams


def train_model(model, optimizer, hyp_optimizer, hyperparams, dataloaders_dict, criterion, patience, initial_bias, input_size, batch_size, n_tokens=None, AttNet=None, ANoptimizer=None):   
    since = time.time()
    val_loss_history = []
    best_f1 = 0.0
    best_f1_acc = 0.0
    
    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts['module.fc.bias'] = initial_bias

    epoch = 0
    #while patience >= 0: # Run untill validation loss has not improved for n epochs equal to patience variable and batchsize has decaed to 1
    while patience > 0 and epoch < args.max_epochs:
        print('\nEpoch {}'.format(epoch))
        print('-' * 10) 

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
                       
                       #Get predictionas from regular or quantised model
                        if args.quantise == True and phase == 'val':
                            outputs = quantized_model(inputs)
                        else:
                            outputs = model(inputs)

                        #use PCA to reduce dimensionality of output to 32x2
                        if args.split_image == True:
                            #old_output_shell = outputs[0:batch_size,0:num_classes]
                            new_batch = []
                            for i in range(batch_size):
                                outputs_ = outputs[i*n_tokens**2:(i+1)*n_tokens**2]

                                #Use PCA to reduce dimensionality of output to 32x2
                                # X = outputs_.detach().cpu().numpy()
                                # pca = PCA(n_components=num_classes)
                                # pca.fit(X)
                                # outputs_ = torch.from_numpy(pca.singular_values_).to(device)
                                # new_batch.append(outputs_)
                                
                                #Take output of token with highest probability of disease (binary)
                                #max_index = torch.argmax(outputs_[:,0])
                                #outputs_ = outputs_[max_index.item()]
                                
                                #flatten outputs_
                                #outputs_ = outputs_.view(-1)
                                #unsqueeze outputs_

                                #outputs to cpu
                                outputs_ = outputs_.detach().cpu()
                                
                                outputs_flat = torch.empty(0)
                                for i in range(outputs_.shape[1]):
                                    outputs_flat = torch.cat((outputs_flat,outputs_[:,i]),0)

                                outputs_ = outputs_flat.to(device).unsqueeze(0)
                                outputs_ = AttNet(outputs_)
                                

                                #Take mean of all outputs
                                #outputs_ = outputs_.mean(dim=0)
                                
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
                            if args.quantise == True:
                                if epoch > 3:
                                    model.apply(torch.quantization.disable_observer)
                                if epoch >= 3:
                                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                   

                   #Calculate statistics
                   #Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
                   #running totals for these metrics by the total number of training or validation samples. This controls for 
                   #the effect of batch size and the fact that the size of the last batch will less than args.batch_size
                    running_loss += loss.item() * args.batch_size # inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data) 
                    running_precision += stats_out['precision'] * args.batch_size # inputs.size(0)
                    running_recall += stats_out['recall'] * args.batch_size # inputs.size(0)
                    running_f1 += stats_out['f1-score'] * args.batch_size # inputs.size(0)    

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
                model_out = model
               # Save only the model weights for easy loading into a new model
                final_out = {
                    'model': best_model_wts,
                    '__author__': 'Jamie R. Sykes',
                    '__model_name__': args.model_name,
                    '__model_parameters__': args                    
                    }       
                
                if args.sweep == True:
                    PATH = os.path.join(args.root, 'models', args.model_name + '_' + wandb.run.name)
                else:
                    PATH = os.path.join(args.root, 'models', args.model_name)
                   
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
                wandb.log({"epoch": epoch, "Train_loss": epoch_loss, "Train_acc": epoch_acc, "Train_F1": epoch_f1,
                           "hyp_1_lr": optimizer.param_groups[0]['lr'], "hyp_2_lr": hyp_optimizer.param_groups[0]['lr'],
                           "hyp_1_eps": optimizer.param_groups[0]['eps'], "hyp_2_eps": hyp_optimizer.param_groups[0]['eps'],
                            "hyp_1_weight_decay": optimizer.param_groups[0]['weight_decay'], "hyp_2_weight_decay": hyp_optimizer.param_groups[0]['weight_decay'],
                           })  
            else:
                wandb.log({"epoch": epoch, "Val_loss": epoch_loss, "Val_acc": epoch_acc, "Val_F1": epoch_f1, "Best_F1": best_f1, "Best_F1_acc": best_f1_acc})

        hyp_optimizer.zero_grad()
        loss = epoch_loss
        loss.backward()
        hyp_optimizer.step()

        optimizer = torch.optim.Adam(model.parameters(),
                               lr=hyp_optimizer.param_groups[0]['lr'], weight_decay=hyp_optimizer.param_groups[0]['weight_decay'], eps=hyp_optimizer.param_groups[0]['eps'])
          

        bar.finish() 
        epoch += 1
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Acc of saved model: {:4f}'.format(best_f1_acc))
    print('F1 of saved model: {:4f}'.format(best_f1))
    return model_out


def build_optimizer(model, optimizer, learning_rate, eps, weight_decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay, eps=eps)
    return optimizer


def build_datasets(input_size, data_dir):
    # Data augmentation and normalization for training
    # Just normalization for device
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomCrop(input_size, pad_if_needed=True, padding_mode = 'reflect'),
            transforms.Resize((input_size,input_size)),
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
    if args.cont_train == True and os.path.exists(os.path.join(args.root, 'models', args.model_name + '.pkl')) == True:
        print('Loading checkpoint weights')
        pretrained_model_wts = pickle.load(open(os.path.join(args.root, 'models', args.model_name + '.pkl'), "rb"))
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
        else:
            print("Architecture name not recognised")
            exit(0)
    # Load custom pretrained weights    

    else:
        print('\nLoading custom pre-trained weights with: ')
        pretrained_model_wts = pickle.load(open(os.path.join(args.root, 'models', args.custom_pretrained_weights), "rb"))
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
    def __init__(self, alpha=1, gamma=2):
        super(DynamicFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights_dict = {}

    def forward(self, inputs, targets):
        logp = nn.functional.log_softmax(inputs, dim=1)
        targets_onehot = nn.functional.one_hot(targets, num_classes=inputs.size(-1))
        targets_onehot = targets_onehot.float()
        pt = torch.exp(logp) * targets_onehot
        loss = -self.alpha * (1 - pt)**self.gamma * logp
        loss = loss.mean()

        # Update weights_dict based on targets and predictions
        preds = torch.argmax(inputs, dim=1)
        for i in range(inputs.size(0)):
            filename = targets[i]
            if filename not in self.weights_dict:
                self.weights_dict[filename] = 1
            if preds[i] != targets[i]:
                self.weights_dict[filename] += 1
            #else:
             #   self.weights_dict[filename] = 1

        # Apply weights to loss based on weights_dict
        weighted_loss = torch.zeros(1).to(loss.device)
        for filename, weight in self.weights_dict.items():
            if weight > 1:
                weighted_loss += loss * weight
            else:
                weighted_loss += loss
        weighted_loss /= len(self.weights_dict)

        return weighted_loss

def sweep_train():

    # Initialize a new wandb run
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller

    run = wandb.init(config=sweep_config)
    data_dir, num_classes, initial_bias, criterion = setup(args)

    model_ft = build_model(num_classes=num_classes)
    
    model_ft = set_batchnorm_momentum(model_ft, momentum=0.001)
    model_ft = model_ft.to(device)  
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-5,
                                           weight_decay=0, eps=1e-8)
    
    #optimizer = build_optimizer(model=model_ft, optimizer=wandb.config.optimizer, learning_rate=1e-5, eps=1e-8, weight_decay=0)
    image_datasets = build_datasets(input_size=int(wandb.config.input_size), data_dir=data_dir)
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()-2, drop_last=True) for x in ['train', 'val']}


    if args.split_image == True:
        AttNet = AttentionNet(num_classes=num_classes, num_tokens=wandb.config.n_tokens)
        AttNet = AttNet.to(device)
        ANoptimizer = torch.optim.Adam(AttNet.parameters(), lr=1e-5, weight_decay=0, eps=1e-8)
        train_model(model=model_ft, optimizer=optimizer, dataloaders_dict=dataloaders_dict, criterion=criterion, patience=args.patience, initial_bias=initial_bias, input_size=int(wandb.config.input_size), batch_size=args.batch_size, AttNet=AttNet, ANoptimizer=ANoptimizer)

    
    train_model(model=model_ft, optimizer=optimizer, dataloaders_dict=dataloaders_dict, criterion=criterion, patience=args.patience, initial_bias=initial_bias, input_size=int(wandb.config.input_size), batch_size=args.batch_size)

def train(args_override=None):
    #if args_override is not None:
       # args = args_override


    wandb.init(project=args.project_name)
    data_dir, num_classes, initial_bias, criterion, hyperparams = setup(args)

    num_classes = len(os.listdir(os.path.join(data_dir, 'train')))
    model_ft = build_model(num_classes=num_classes)
    model_ft = set_batchnorm_momentum(model_ft, momentum=0.001)

    model_ft = model_ft.to(device)
    # optimizer = torch.optim.Adam(model_ft.parameters(), lr=hyper_optimizer['learning_rate'],
    #                                        weight_decay=hyper_optimizer['weight_decay'], eps=hyper_optimizer['eps'])
    
    
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=hyperparams['learning_rate'],
                                           weight_decay=hyperparams['weight_decay'], eps=hyperparams['eps'])
    
    hyp_optimizer = torch.optim.SGD(hyperparams.values(), lr=1e-5, momentum=0.9, weight_decay=0)
            

    image_datasets = build_datasets(kernel_size=5, sigma_max=2, input_size=args.input_size, data_dir=data_dir)
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()-2, drop_last=True) for x in ['train', 'val']}

    # weights_dict = {}
    # for i in range(len(image_datasets['train'])):
    #     image_path, _ = image_datasets['train'].samples[i]
    #     weights_dict[os.path.basename(image_path)] = 1.0

    if args.split_image == True:
        AttNet = AttentionNet(num_classes=num_classes, num_tokens=args.n_tokens)
        AttNet = AttNet.to(device)
        ANoptimizer = torch.optim.Adam(AttNet.parameters(),
                               lr=1e-5, weight_decay=0, eps=1e-8)
    
        model_out = train_model(model=model_ft, optimizer=optimizer, hyp_optimizer=hyp_optimizer, hyperparams=hyperparams, dataloaders_dict=dataloaders_dict, criterion=criterion, patience=args.patience, initial_bias=initial_bias, input_size=args.input_size, n_tokens=args.n_tokens, batch_size=args.batch_size, AttNet=AttNet, ANoptimizer=ANoptimizer)

    model_out = train_model(model=model_ft, optimizer=optimizer, hyp_optimizer=hyp_optimizer, hyperparams=hyperparams, dataloaders_dict=dataloaders_dict, criterion=criterion, patience=args.patience, initial_bias=initial_bias, input_size=args.input_size, n_tokens=args.n_tokens, batch_size=args.batch_size, AttNet=None, ANoptimizer=None)
    
    return model_out, image_datasets



if args.sweep == True:
    with open(args.sweep_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        sweep_config = config['sweep_config']
        sweep_config['metric'] = config['metric']
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
    
    wandb.agent(sweep_id,
            project=args.project_name, 
            function=sweep_train,
            count=args.sweep_count)
else:
    train()