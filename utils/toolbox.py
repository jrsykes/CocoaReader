from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from ArchitectureZoo import DisNetV1_2, PhytNet_SRAutoencoder
import timm
from thop import profile
from sklearn.metrics import f1_score
from itertools import combinations
# import torch.nn.functional as F
from torch.utils.data import Sampler
import pandas as pd
from ete3 import Tree
from Bio import Phylo
from io import StringIO

def build_model(num_classes, arch, config):
    print()
    print('Building model...')
 
    if arch == 'convnext_tiny':
        print('Loaded ConvNext Tiny with pretrained IN weights')
        model_ft = models.convnext_tiny(weights = None)
        in_feat = model_ft.classifier[2].in_features
        model_ft.classifier[2] = torch.nn.Linear(in_feat, num_classes)
    elif arch == 'resnet18':
        model_ft = models.resnet18(weights=None)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif arch == 'resnet50':
        model_ft = models.resnet50(weights=None)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif arch == 'PhytNet_SRAutoencoder':
        model_ft = PhytNet_SRAutoencoder(config=config)
    elif arch == 'DisNet_SRAutoencoder':
        model_ft = DisNet_SRAutoencoder(config=config)
    elif arch == 'DisNetV1_2':
        model_ft = DisNetV1_2(config=config)
    elif arch == 'efficientnetv2_s':
        model_ft = timm.create_model('tf_efficientnetv2_s', pretrained=False)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
    elif arch == 'efficientnet_b0':
        model_ft = timm.create_model('efficientnet_b0', pretrained=False)
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
    try:
        num_classes= len(os.listdir(data_dir + '/train'))
        dir_ = os.path.join(data_dir, 'train')
    except:
        num_classes= len(os.listdir(data_dir + '/fold_0/train'))
        dir_ = os.path.join(data_dir, 'fold_0/train')

    # Specify whether to use GPU or cpu. Quantisation aware training is not yet avalable for GPU.
   
    #device = torch.device("cuda")
    device = torch.device("cuda:" + args.GPU)


    ### Calculate and set bias for final layer based on imbalance in dataset classes
    
    # list_cats = []
    # for i in sorted(os.listdir(dir_)):
    #     _, _, files = next(os.walk(os.path.join(dir_, i)))
    #     list_cats.append(len(files))

    # weights = []
    # for i in list_cats:
    #     weights.append(np.log((max(list_cats)/i)))

    # initial_bias = torch.FloatTensor(weights)

    return data_dir, num_classes, device

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

class AveragePoolingColorChannels(object):
    def __call__(self, img):
        # Compute the mean along the color channel, then add an extra color channel dimension
        return torch.mean(img, dim=0, keepdim=True)

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
                transforms.ToTensor()

            ]),
            'val': transforms.Compose([
                transforms.ToTensor()
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
                transforms.ToTensor(),
                #AveragePoolingColorChannels()

            ]),
            'val': transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.ToTensor(),
                #AveragePoolingColorChannels()
            ])
        }   

    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    return image_datasets


def SetSeeds(seed=42):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    


class Metrics:
    def __init__(self, metric_names, num_classes):
        if metric_names == "All":
            metric_names = ['loss', 'corrects', 'precision', 'recall', 'f1']
        self.metric_names = metric_names
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.metrics = {
            'loss': 0.0,
            'cont_loss': 0.0,
            'Genetic_loss': 0.0,
            'MSE_loss': 0.0,
            'corrects': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        self.n = 0
        self.all_preds = []
        self.all_labels = []

    def update(self, loss=None, cont_loss=None, Genetic_loss=None, MSE_loss=None, preds=None, labels=None, stats_out=None):
        inputs_size = labels.size(0)
        if 'loss' in self.metric_names:
            self.metrics['loss'] += loss.item() * inputs_size
        if 'cont_loss' in self.metric_names:
            self.metrics['cont_loss'] += cont_loss.item() * inputs_size
            
        if 'Genetic_loss' in self.metric_names:
            self.metrics['Genetic_loss'] += Genetic_loss.item() * inputs_size
            
        if 'MSE_loss' in self.metric_names:
            self.metrics['MSE_loss'] += MSE_loss.item() * inputs_size
        if 'corrects' in self.metric_names:
            self.metrics['corrects'] += torch.sum(preds == labels.data)
        if 'precision' in self.metric_names:
            self.metrics['precision'] += stats_out['precision'] * inputs_size
        if 'recall' in self.metric_names:
            self.metrics['recall'] += stats_out['recall'] * inputs_size
        if 'f1' in self.metric_names:
            self.metrics['f1'] += stats_out['f1-score'] * inputs_size
        self.n += inputs_size

        if preds != None:
            # Store all predictions and labels for later calculation
            self.all_preds.extend(preds.cpu().numpy())
            self.all_labels.extend(labels.cpu().numpy())

    def calculate(self):
        results = {}
        if 'loss' in self.metric_names:
            results['loss'] = self.metrics['loss'] / self.n
        if 'cont_loss' in self.metric_names:
            results['cont_loss'] = self.metrics['cont_loss'] / self.n
        if 'Genetic_loss' in self.metric_names:
            results['Genetic_loss'] = self.metrics['Genetic_loss'] / self.n
        if 'MSE_loss' in self.metric_names:
            results['MSE_loss'] = self.metrics['MSE_loss'] / self.n
        if 'corrects' in self.metric_names:
            results['acc'] = self.metrics['corrects'].double() / self.n
        if 'precision' in self.metric_names:
            results['precision'] = self.metrics['precision'] / self.n
        if 'recall' in self.metric_names:
            results['recall'] = self.metrics['recall'] / self.n
        if 'f1' in self.metric_names:
            results['f1'] = self.metrics['f1'] / self.n
            results['f1_per_class'] = f1_score(self.all_labels, self.all_preds, average=None)

        return results



def count_flops(model, device, input_size):
    inputs = torch.randn(1, input_size[0], input_size[1], input_size[2]).to(device)
    flops, params = profile(model, inputs=(inputs, ), verbose=False)

    # Convert to GFLOPs
    GFLOPs = flops / 1e9
    return GFLOPs, params





def contrastive_loss_with_dynamic_margin(encoded, distances, labels):
    class_list = distances.index.tolist()
    encoded_images_lst = [(enc, class_list[label]) for enc, label in zip(encoded, labels)]

    pairs = list(combinations(encoded_images_lst, 2))

    loss = 0.0
    alpha = 7 #Weigths the relative importance of genetic distance vs euclidian distance. Higher = euclid, lower = genetic
    beta = distances.values.max() #A constant forcing all values to be positive

    for (encoded1, class1), (encoded2, class2) in pairs:
        margin = distances.loc[class1][class2]
        euclidean_distance = torch.norm(encoded1 - encoded2)

        loss += (euclidean_distance*alpha-margin)+beta 

    return loss




class NineImageSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def get_or_create_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        new_child = Node(name)
        self.children.append(new_child)
        return new_child

    def to_ete(self):
        t = Tree()
        t.name = self.name
        for child in self.children:
            t.add_child(child.to_ete())
        return t

def generate_newick(data):
    root = Node('Root')

    # Traverse the DataFrame to build the tree structure
    for _, row in data.iterrows():
        current = root
        for col in data.columns[1:-1]:
            taxon = row[col]
            if pd.notna(taxon):
                current = current.get_or_create_child(taxon)


    # Convert to ete3 tree
    ete_tree = root.to_ete()

    newick_str = ete_tree.write(format=9)  # format=1 includes branch lengths
    
    return newick_str




def traverse_and_tokenize(tree):
    tokens = []

    def preorder(node):
        if node.is_terminal():  # It's a leaf node
            tokens.append(node.name)
        else:
            tokens.append('(')
            for idx, child in enumerate(node.clades):
                preorder(child)
                if idx < len(node.clades) - 1:
                    tokens.append(',')
            tokens.append(')')

    preorder(tree.root)
    return tokens


def generate_label_relationship_matrix(tree):
    # Get all terminal (leaf) node names
    terminals = tree.get_terminals()
    terminal_names = [terminal.name for terminal in terminals]
    
    # Initialize an empty matrix
    num_terminals = len(terminal_names)
    label_relationship_matrix = np.zeros((num_terminals, num_terminals))
    
    # Fill the matrix with distances
    for i, terminal_i in enumerate(terminals):
        for j, terminal_j in enumerate(terminals):
            # Calculate the distance between each pair of terminals
            distance = tree.distance(terminal_i, terminal_j)
            label_relationship_matrix[i, j] = distance
    
    return label_relationship_matrix