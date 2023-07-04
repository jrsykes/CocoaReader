from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from ArchitectureZoo import DisNet_pico, DisNet_pico_deep, DisNet_nano
import timm
from thop import profile

def build_model(num_classes, arch, config):
    print()
    print('Building model...')
 
    if arch == 'convnext_tiny':
        print('Loaded ConvNext Tiny with pretrained IN weights')
        model_ft = models.convnext_tiny(weights = True)
        in_feat = model_ft.classifier[2].in_features
        model_ft.classifier[2] = torch.nn.Linear(in_feat, num_classes)
    elif arch == 'resnet18':
        model_ft = models.resnet18(weights=True)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif arch == 'resnet50':
        model_ft = models.resnet50(weights=None)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif arch == 'DisNet-pico':
        model_ft = DisNet_pico(out_channels=num_classes, config_dict=config)
    elif arch == 'DisNet-pico_deep':
        model_ft = DisNet_pico_deep(out_channels=num_classes, config_dict=config)

    elif arch == 'DisNet-nano':
        model_ft = DisNet_nano(out_channels=num_classes)
        
    elif arch == 'efficientnetv2_s':
        model_ft = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
    elif arch == 'efficientnet_b0':
        model_ft = timm.create_model('efficientnet_b0', pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
                       
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
    print("Setting stuff up...")
    print('#'*50)
    data_dir = os.path.join(args.root, args.data_dir)
    #Define some variable and paths
    os.environ['TORCH_HOME'] = os.path.join(args.root, "TORCH_HOME")
    
    data_dir = os.path.join(args.root, args.data_dir)
    num_classes= len(os.listdir(data_dir + '/train'))

    # Specify whether to use GPU or cpu. Quantisation aware training is not yet avalable for GPU.
   
    #device = torch.device("cuda")
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


    ### Calculate and set bias for final layer based on imbalance in dataset classes
    dir_ = os.path.join(data_dir, 'train')
    list_cats = []
    for i in sorted(os.listdir(dir_)):
        _, _, files = next(os.walk(os.path.join(dir_, i)))
        list_cats.append(len(files))

    weights = []
    for i in list_cats:
        weights.append(np.log((max(list_cats)/i)))

    initial_bias = torch.FloatTensor(weights)

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
    print("Image input size: ", str(input_size))
    if input_size == None:
        print("No image resize applied")
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(1,3)),
                                        transforms.RandomRotation(degrees=5)
                                        ], p=0.4), 
                transforms.ToTensor(),

            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ])
        }

    elif input_size != None:
        print("Image resize applied")
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size,input_size)), 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(1,3)),
                        transforms.RandomRotation(degrees=5)
                        ], p=0.4), 
                transforms.ToTensor()

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
    


class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.running_loss = 0.0
        self.running_corrects = 0
        self.running_precision = 0.0
        self.running_recall = 0.0
        self.running_f1 = 0.0
        self.n = 0

    def update(self, loss, preds, labels, stats_out):
        inputs_size = labels.size(0)
        self.running_loss += loss.item() * inputs_size
        self.running_corrects += torch.sum(preds == labels.data)
        self.running_precision += stats_out['precision'] * inputs_size
        self.running_recall += stats_out['recall'] * inputs_size
        self.running_f1 += stats_out['f1-score'] * inputs_size
        self.n += inputs_size

    def calculate(self):
        loss = self.running_loss / self.n
        acc = self.running_corrects.double() / self.n
        precision = self.running_precision / self.n
        recall = self.running_recall / self.n
        f1 = self.running_f1 / self.n
        return loss, acc, precision, recall, f1

def count_flops(model, device):
    inputs = torch.randn(1, 3, 400, 400).to(device)
    flops, params = profile(model, inputs=(inputs, ), verbose=False)

    # Convert to GFLOPs
    GFLOPs = flops / 1e9
    return GFLOPs, params

