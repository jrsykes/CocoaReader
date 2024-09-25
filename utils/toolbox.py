from __future__ import print_function
from __future__ import division

import os
from typing import Any
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from ArchitectureZoo import PhytNetV0, PhytNet_SRAutoencoder
from PhytNet_ablation import PhytNetV0_ablation
import timm
from thop import profile
from sklearn.metrics import f1_score
# from itertools import combinations
import torch.nn.functional as F
from torch.utils.data import Sampler
import networkx as nx


def build_model(num_classes, arch, config):
    print()
    print('Building model...')
 
    if arch == 'convnext_tiny':
        print('Loaded ConvNext Tiny with pretrained IN weights')
        model_ft = models.convnext_tiny(weights = None)
        in_feat = model_ft.classifier[2].in_features
        model_ft.classifier[2] = torch.nn.Linear(in_feat, num_classes)
    elif arch == 'resnet18':
        model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif arch == 'resnet50':
        model_ft = models.resnet50(weights=None)
        in_feat = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_feat, num_classes)
    elif arch == 'PhytNet_SRAutoencoder':
        model_ft = PhytNet_SRAutoencoder(config=config)
    elif arch == 'PhytNetV0':
        model_ft = PhytNetV0(config=config)
    elif arch == 'PhytNetV0_ablation':
        model_ft = PhytNetV0_ablation(config=config)
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
        if step > len(self.dataloader) * inputs.size(0): #wait until after first epoch to start updating weights_dict
            ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss

            # Update weights_dict based on targets and predictions
            preds = torch.argmax(inputs, dim=1)
            gamma = 0
            for i in range(inputs.size(0)):
                #get filename from dataset
                filename = self.dataloader.dataset.samples[step + i][0].split("/")[-1]
                if filename not in self.weights_dict:
                    self.weights_dict[filename] = 1
                if preds[i] != targets[i]:
                    self.weights_dict[filename] += self.delta

                weight = self.weights_dict[filename]
                if weight > 1:
                    gamma += weight

            #probability of a sample being correctly classified
            pt = torch.exp(-ce_loss)
            #probability of a sample being incorrectly classified multiplied by the weight gamma
            pt_weighted = ((1-pt)**gamma)
            #tensore of 1s and 0s, 1 if the sample is incorrectly classified
            true_incorrects = 1-(torch.argmax(inputs, dim=1) == targets).int()
            
            DFLoss = ((1 + pt_weighted * true_incorrects) * ce_loss).mean()

        else:
            DFLoss = nn.CrossEntropyLoss()(inputs, targets)
            
        step += inputs.size(0)
        return DFLoss, step
    
class Soft_max_focal_loss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(Soft_max_focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, outputs, targets):
    
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        
        pt = torch.exp(-ce_loss)

        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        return focal_loss
    
    
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
            metric_names = ['loss', 'corrects', 'precision', 'recall', 'f1', 'L1']
        self.metric_names = metric_names
        print("\nMetrics to be calculated: ", self.metric_names)
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.metrics = {
            'loss': 0.0,
            'cont_loss': 0.0,
            'Genetic_loss': 0.0,
            'SR_loss': 0.0,
            'Euclid_loss': 0.0,
            'RF_loss': 0.0,
            'ESS': 0.0,
            'L1': 0.0,
            'corrects': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        self.n = 0
        self.all_preds = []
        self.all_labels = []

    def update(self, loss=None, cont_loss=None, Genetic_loss=None, SR_loss=None, 
               Euclid_loss=None, RF_loss=None, ESS=None, L1=None,
               preds=None, labels=None, stats_out=None):
        inputs_size = labels.size(0)
        if 'loss' in self.metric_names:
            self.metrics['loss'] += loss.item() * inputs_size
        if 'cont_loss' in self.metric_names:
            self.metrics['cont_loss'] += cont_loss.item() * inputs_size
        if 'Genetic_loss' in self.metric_names:
            self.metrics['Genetic_loss'] += Genetic_loss.item() * inputs_size
        if 'Euclid_loss' in self.metric_names:
            self.metrics['Euclid_loss'] += Euclid_loss.item() * inputs_size
        if 'SR_loss' in self.metric_names:
            self.metrics['SR_loss'] += SR_loss.item() * inputs_size
        if 'RF_loss' in self.metric_names:
            self.metrics['RF_loss'] += RF_loss.item() * inputs_size
        if 'ESS' in self.metric_names:
            self.metrics['ESS'] += ESS.item() * inputs_size  
        if 'L1' in self.metric_names:
            self.metrics['L1'] += L1.item() * inputs_size  
            
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
        if 'Euclid_loss' in self.metric_names:
            results['Euclid_loss'] = self.metrics['Euclid_loss'] / self.n
        if 'SR_loss' in self.metric_names:
            results['SR_loss'] = self.metrics['SR_loss'] / self.n
        if 'RF_loss' in self.metric_names:
            results['RF_loss'] = self.metrics['RF_loss'] / self.n
        if 'ESS' in self.metric_names:
            results['ESS'] = self.metrics['ESS'] / self.n
        if 'L1' in self.metric_names:
            results['L1'] = self.metrics['L1'] / self.n
            
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


# def generate_random_mask(input_size, mask_ratio=0.6, device='cuda'):
#     """
#     Generates a random binary mask with multiple masked patches.
#     :param input_size: tuple, the size of the input image (C, H, W).
#     :param mask_size: tuple, the size of the masked region (h, w).
#     :param mask_ratio: float, the proportion of the image to be masked.
#     :param device: str, the device to create the mask on.
#     :return: torch.Tensor, a binary mask of size (1, C, H, W).
#     """
#     _, _, H, W = input_size
#     h, w = H // 8, W // 8  # Size of the patches
    
#     # Calculate the number of patches to mask
#     total_patches = (H * W) / (h * w)
#     num_masked_patches = int(total_patches * mask_ratio)
    
#     # Creating the mask
#     mask = torch.ones(*input_size, device=device)
    
#     for _ in range(num_masked_patches):
#         # Starting coordinates for the mask
#         top = torch.randint(0, H - h + 1, (1,)).item()
#         left = torch.randint(0, W - w + 1, (1,)).item()
        
#         mask[:, :, top:top + h, left:left + w] = 0
    
#     return mask




# def contrastive_loss_with_dynamic_margin(encoded, distances, labels):
    
#     class_list = distances.index.tolist()
#     encoded_images_lst = [(enc, class_list[label]) for enc, label in zip(encoded, labels)]

#     pairs = list(combinations(encoded_images_lst, 2))

#     loss, loss2 = 0.0, 0.0
#     # alpha = 10 #Weigths the relative importance of genetic distance vs euclidian distance. Higher = euclid, lower = genetic
#     # beta = distances.values.max() #A constant forcing all values to be positive
#     # beta = 40
    
#     for (encoded1, class1), (encoded2, class2) in pairs:
#         margin = distances.loc[class1][class2]
#         euclidean_distance = torch.norm(encoded1 - encoded2)
        
        
#         # loss += torch.MSELoss(reduction=None)(euclidean_distance, margin)
#         loss += (euclidean_distance*alpha-margin)+beta
#         # loss += (margin-euclidean_distance*alpha)+beta
#         # loss = (margin+euclidean_distance*alpha)+beta
#         # loss2 += torch.exp(euclidean_distance*(1-margin/beta))-1
  
#     return loss/10000



class NineImageSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def lower_triangle(matrix):
    lower = []
    for i in range(len(matrix)):
        row = []
        for j in range(i + 1):
            row.append(matrix[i][j])
        lower.append(row)
    return lower

def create_phylogenetic_dag(data):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges based on the hierarchy
    # clade1 -> clade2 -> clade3 -> order -> family -> subfamily -> genus
    for _, row in data.iterrows():
        G.add_edge(row['clade0'], row['clade1'])
        G.add_edge(row['clade1'], row['clade2'])
        G.add_edge(row['clade2'], row['clade3'])
        G.add_edge(row['clade3'], row['order'])
        G.add_edge(row['order'], row['family'])
        G.add_edge(row['family'], row['subfamily'])
        G.add_edge(row['subfamily'], row['genus'])
        G.add_edge(row['genus'], row['label'])

    # Check if the graph has cycles and convert to DAG if necessary
    while True:
        try:
            # Finds a cycle and returns an iterator that can be used to remove the edge.
            cycle = nx.find_cycle(G, orientation='original')
            # Remove the last edge in the cycle, which should break the cycle
            G.remove_edge(cycle[-1][0], cycle[-1][1])
        except nx.NetworkXNoCycle:
            # If no cycle is found, break the loop
            break
    
    for u, v in G.edges():
        # Calculate path length
        path_length = nx.shortest_path_length(G, source=u, target=v)
        # Apply logarithmic transformation
        G[u][v]['weight'] = np.log(1e-6 + path_length)
        
    return G