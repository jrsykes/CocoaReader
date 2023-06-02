from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from ArchitectureZoo import DisNet_nano, DisNet_pico



def build_model(num_classes, arch, config):
    print()
    print('Building model...')
 
    if arch == 'convnext_tiny':
        print('Loaded ConvNext Tiny with pretrained IN weights')
        model_ft = models.convnext_tiny(weights = None)
        in_feat = model_ft.classifier[2].in_features
        model_ft.classifier[2] = torch.nn.Linear(in_feat, num_classes)
    elif arch == 'resnet18':
        print('Loaded ResNet18 with pretrained IN weights')
        model_ft = models.resnet18(weights=None)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif arch == 'resnet50':
        print('Loaded ResNet50 with pretrained IN weights')
        model_ft = models.resnet50(weights=None)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif arch == 'DisNet_pico':
        print('Loaded DisNet_pico')
        
        model_ft = DisNet_pico(config)
    elif arch == 'DisNet_nano':
        print('Loaded DisNet_nano')
        
        model_ft = DisNet_nano(config)
                       
    else:
        print("Architecture name not recognised")
        exit(0)
    print()
    print(f'{arch} loaded')
    print('#'*50)
    print()
    # Load custom pretrained weights    
    return model_ft

def set_batchnorm_momentum(self, momentum):
    for m in self.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = momentum
    return self

def setup(args):
    data_dir = os.path.join(args.root, args.data_dir)
    #Define some variable and paths
    os.environ['TORCH_HOME'] = os.path.join(args.root, "TORCH_HOME")
    
    data_dir = os.path.join(args.root, args.data_dir)
    num_classes= len(os.listdir(data_dir + '/train'))

    # Specify whether to use GPU or cpu. Quantisation aware training is not yet avalable for GPU.
   
    device = torch.device("cuda")
    
    ### Calculate and set bias for final layer based on imbalance in dataset classes
    dir_ = os.path.join(data_dir, 'train')
    list_cats = []
    for i in sorted(os.listdir(dir_)):
        _, _, files = next(os.walk(os.path.join(dir_, i)))
        list_cats.append(len(files))

    weights = []
    for i in list_cats:
        weights.append(np.log((max(list_cats)/i)))

    initial_bias = torch.FloatTensor(weights).to(device)

    return data_dir, num_classes, initial_bias, device

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class DynamicFocalLoss(nn.Module):
    def __init__(self, delta=1, dataloader=None):
        super(DynamicFocalLoss, self).__init__()
        self.delta = delta
        self.dataloader = dataloader
        self.weights_dict = {}

    def forward(self, inputs, targets, step):
        loss = nn.CrossEntropyLoss()(inputs, targets)
        if step > len(self.dataloader): #wait until after first epoch to start updating weights_dict
            # Update weights_dict based on targets and predictions
            preds = torch.argmax(inputs, dim=1)
            batch_weight = 0
            for i in range(inputs.size(0)):
                #get filename from dataset
                filename = self.dataloader.dataset.samples[step + i][0].split("/")[-1]
                if filename not in self.weights_dict:
                    self.weights_dict[filename] = 1
                if preds[i] != targets[i]:
                    self.weights_dict[filename] += self.delta

                weight = self.weights_dict[filename]
                if weight > 1:
                    batch_weight += weight

            loss *= batch_weight
        step += inputs.size(0)
        
        return loss, step
    
def Remove_module_from_layers(unpickled_model_wts):
    new_keys = []
    for key, value in unpickled_model_wts.items():
        new_keys.append(key.replace('module.', ''))
    for i in new_keys:
        unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)
    return unpickled_model_wts

def AIC(model, loss):
    #get number of model peramaters from the model
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)   

    AIC_ = 2*k - 2*np.log(loss)
    return AIC_

def build_datasets(data_dir, input_size=None):
    # Data augmentation and normalization for training
    # Just normalization for device
    if input_size == None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),

            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ])
        }

    elif input_size != None:
        data_transforms = {
            'train': transforms.Compose([
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


def SetSeeds():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)