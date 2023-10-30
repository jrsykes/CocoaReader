from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from ArchitectureZoo import DisNetV1_2, DisNet_SRAutoencoder
import timm
from thop import profile
from sklearn.metrics import f1_score
from itertools import combinations
import torch.nn.functional as F
from torch.utils.data import Sampler


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
    # elif arch == 'DisNet_MaskedAutoencoder':
    #     model_ft = DisNet_MaskedAutoencoder(config=config)
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
            'MSE_loss': 0.0,
            'corrects': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        self.n = 0
        self.all_preds = []
        self.all_labels = []

    def update(self, loss=None, cont_loss=None, MSE_loss=None, preds=None, labels=None, stats_out=None):
        inputs_size = labels.size(0)
        if 'loss' in self.metric_names:
            self.metrics['loss'] += loss.item() * inputs_size
        if 'cont_loss' in self.metric_names:
            self.metrics['cont_loss'] += cont_loss.item() * inputs_size
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




def contrastive_loss_with_dynamic_margin(encoded, distances, labels):
    
    class_list = distances.index.tolist()
    max_distance = distances.max().max()
    
    encoded_images_lst = [(enc, class_list[label]) for enc, label in zip(encoded, labels)]

    pairs = list(combinations(encoded_images_lst, 2))

    num_pairs = len(pairs)
    loss = 0.0

    for (encoded1, class1), (encoded2, class2) in pairs:
        margin = distances.loc[class1][class2] / max_distance
        euclidean_distance = torch.norm(encoded1 - encoded2)
 
        if class1 == class2:
            y_true = 1
        else:
            y_true = 0
        
        loss += (1 - y_true) * 0.5 * euclidean_distance**2 + y_true * 0.5 * (torch.clamp(margin - euclidean_distance, min=0.0))**2


    return loss / num_pairs



def compute_combined_gradients(model, optimizer, losses):
    """
    Compute combined gradients for a list of losses.

    Parameters:
    - model: The model for which gradients are computed.
    - optimizer: The optimizer used to zero out gradients.
    - losses: A list of losses.

    Returns:
    - combined_grads: A list of combined gradients for each model parameter.
    """
    
    all_grads = []

    for loss in losses:
        # Zero the gradients
        optimizer.zero_grad()

        # Compute gradients for the current loss
        loss.backward(retain_graph=True)
        current_grads = []
        for param in model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.data.clone())
            else:
                current_grads.append(None)
        all_grads.append(current_grads)

    # Normalize the gradients for each loss
    normalized_all_grads = []
    for grads in all_grads:
        norm = torch.sqrt(sum(torch.sum((g ** 2)) for g in grads if g is not None) + torch.tensor(0.0))
        normalized_grads = [g / norm if g is not None else None for g in grads]
        normalized_all_grads.append(normalized_grads)

    # Combine the normalized gradients
    num_losses = len(losses)
    combined_grads = []
    for grads in zip(*normalized_all_grads):
        valid_grads = [g for g in grads if g is not None]
        combined_grad = sum(valid_grads) / num_losses
        combined_grads.append(combined_grad)

    # Manually update the model parameters using the combined gradients
    for param, grad in zip(model.parameters(), combined_grads):
        if grad is not None and param.grad is not None:
            param.grad.data.copy_(grad)
            param.grad = None
    


class NineImageSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
