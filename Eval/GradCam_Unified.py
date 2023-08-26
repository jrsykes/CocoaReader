#%%
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append('~/scripts/CocoaReader/utils/')

import toolbox

#%%
class ResNet_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(ResNet_CAM, self).__init__()
        self.resnet = net
        convs = nn.Sequential(*list(net.children())[:-1])
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(*list(net.children())[-1:])
        
    def forward(self, x):
        x = self.first_part_conv(x.to(device))
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view((1, -1))
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x.to(device))
    
#%%
class DisNet_pico_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(DisNet_pico_CAM, self).__init__()
        self.disnet_pico = net
        self.conv1 = net.conv1  # Define these layers individually
        self.cnblock1 = net.cnblock1
        self.se1 = net.se1
        self.Maxpool = net.Maxpool
        self.conv2 = net.conv2
        self.cnblock2 = net.cnblock2
        self.Avgpool = net.Avgpool
        self.fc1 = net.fc1
        self.fc2 = net.fc2
        self.fc2_norm = net.fc2_norm
        self.fc3 = net.fc3
        
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))  # Apply relu here
        x = self.cnblock1(x)
        x = self.se1(x)
        x = self.Maxpool(x)
        x = F.relu(self.conv2(x))  # Apply relu here
        x = self.cnblock2(x)
        x = self.Avgpool(x)
        x.register_hook(self.activations_hook)
        x = torch.flatten(x, 1) # flatten before feeding to linear layers
        x = F.gelu(self.fc1(x))     
        x = F.gelu(self.fc2(x)) 
        x = self.fc2_norm(x)    
        x = self.fc3(x) 
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))  # Apply relu here
        x = self.cnblock1(x)
        x = self.se1(x)
        x = self.Maxpool(x)
        x = F.relu(self.conv2(x))  # Apply relu here
        x = self.cnblock2(x)
        x = self.Avgpool(x)
        return x


# %%

def superimpose_heatmap(heatmap, img):
    resized_heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))

    # Replace NaN or Inf with 0
    resized_heatmap = np.nan_to_num(resized_heatmap, nan=0, posinf=0, neginf=0)

    # Normalizing the heatmap between 0 and 1 for eliminating any invalid values
    min_val = np.min(resized_heatmap)
    max_val = np.max(resized_heatmap)
    if max_val > min_val:  # To avoid division by zero
        resized_heatmap = (resized_heatmap - min_val) / (max_val - min_val)
    else:
        resized_heatmap = np.zeros_like(resized_heatmap)

    resized_heatmap = np.uint8(255 * resized_heatmap)  # Now it can be correctly casted to uint8
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)

    # Converting image to the same data type as superimposed_img
    img_as_float = img[0].permute(1,2,0).numpy().astype(float)

    superimposed_img = cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB) * 0.006 + img_as_float
    superimposed_img = torch.from_numpy(superimposed_img)  # Convert back to tensor if necessary

    return superimposed_img



def get_grad_cam(net, img):
    net.eval()
    pred = net(img)
    pred[:,pred.argmax(dim=1)].backward()
    gradients = net.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = net.get_activations(img).detach()
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze().detach().to('cpu')
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    
    #clearning memory
    del gradients, pooled_gradients, activations, pred
    
    return torch.Tensor(superimpose_heatmap(heatmap, img).permute(2,0,1))
#%%

config = {
    'num_classes': 8,
    'drop_out': 0.3,
    'drop_out2': 0.3,
    "dim_1": 26,
    "dim_2": 16,
    "input_size": 300,
    "kernel_1": 1,
    "kernel_2": 1,
    "kernel_3": 9,
    "kernel_4": 3,
    "kernel_5": 19,
    "kernel_6": 11,
    "nodes_1": 168,
    "nodes_2": 88,
    "nodes_3": 79,
    "nodes_4": 195,
    "seed": 65,
    "trans_nodes": 177}

toolbox.SetSeeds(config['seed'])

DisNet = toolbox.build_model(num_classes=config['trans_nodes'], arch='DisNet_picoIR', config=config)
SecondNet = toolbox.build_model(num_classes=config['trans_nodes'], arch='resnet18', config=None)
Meta = toolbox.build_model(num_classes=config['num_classes'], arch='Meta', config=config)

device = torch.device("cuda:0")

config2 = {'CNN1': DisNet, 
           'CNN2': SecondNet, 
           'MetaModel': Meta}
model = toolbox.build_model(num_classes=None, arch='Unified', config=config2).to(device)



#%%
model_set = [(DisNet, DisNet_pico_CAM, 300), (ResNet, ResNet_CAM, 408)]

for i in range(len(model_set)):
    img = torch.randn(1, 3, model_set[i][2], model_set[i][2], requires_grad=True)
    out = get_grad_cam(model_set[i][1](model_set[i][0], -1), img)

