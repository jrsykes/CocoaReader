#%%
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
#import models
#from torchbearer import Trial
import cv2
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
import sys
sys.path.append('/users/jrs596/scripts/CocoaReader/utils')
from ArchitectureZoo import DisNet_pico
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ArchitectureZoo import DisNet_pico, DisNet_pico_deep, DisNet_nano



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
        x = self.first_part_conv(x.to('cuda'))
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
        return self.first_part_conv(x.to('cuda'))
    
#%%


class EfficientNet_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(EfficientNet_CAM, self).__init__()
        self.efficientnet = net
        convs = nn.Sequential(*list(net.children())[:-1])
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:-1]  # Exclude the last layer
        self.linear = nn.Sequential(convs[-1], *list(net.children())[-1:])  # Include the last layer in linear

        
    def forward(self, x):
        x = self.first_part_conv(x.to('cuda'))
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = F.adaptive_avg_pool2d(x, (1280, 1280))
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x.to('cuda'))


#%%

class DisNet_pico_deep_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(DisNet_pico_deep_CAM, self).__init__()
        self.disnet_pico = net
        convs = nn.Sequential(net.conv1, net.cnblock1, net.conv2, net.cnblock2, net.pool, net.conv3, net.cnblock3,
                              net.conv4, net.cnblock4, net.pool)
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(net.fc1, net.fc2, net.fc3)
        
    def forward(self, x):
        x = self.first_part_conv(x.to('cuda'))
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = torch.flatten(x, 1) # flatten before feeding to linear layers
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x.to('cuda'))

class DisNet_pico_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(DisNet_pico_CAM, self).__init__()
        self.disnet_pico = net
        convs = nn.Sequential(net.conv1, net.cnblock1, net.pool, net.conv2, net.cnblock2, net.pool)
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(net.fc1, net.fc2, net.fc3)
        
    def forward(self, x):
        x = self.first_part_conv(x.to('cuda'))
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = torch.flatten(x, 1) # flatten before feeding to linear layers
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x.to('cuda'))

# %%

def superimpose_heatmap(heatmap, img):
    resized_heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))
    resized_heatmap = np.uint8(255 * resized_heatmap)
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    superimposed_img = torch.Tensor(cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB)) * 0.006 + img[0].permute(1,2,0)
    
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
# Load models
resnet = torch.load('/users/jrs596/scratch/dat/models/resnet18-IR.pth')
#DisNet_pico = torch.load('/users/jrs596/scratch/dat/models/DisNet_pico_IR.pth', map_location=lambda storage, loc: storage.cuda(0))
DisNet_pico_DN = torch.load('/users/jrs596/scratch/dat/models/DisNet_pico_IR_DN.pth', map_location=lambda storage, loc: storage.cuda(0))
#DisNet_pico_deep = torch.load('/users/jrs596/scratch/dat/models/DisNet_pico_deep_IR.pth', map_location=lambda storage, loc: storage.cuda(0))


config = {"dim_1":16,
        "dim_2":15,
        "dropout":0.31265223232454825,
        "kernel_1":7,
        "kernel_2":3,
        "kernel_3":4,
        "kernel_4":5,
        "kernel_5":4,
        "kernel_6":6,
        "nodes_1":74,
        "nodes_2":64}

DisNet_pico = DisNet_pico(out_channels=4, config_dict=config).to('cuda')

# Load weights
weights_path = "/users/jrs596/scratch/dat/models/DisNet_pico_IR_weights.pth"
weights = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(0))

# Apply weights to the model
DisNet_pico.load_state_dict(weights)

efficientnet = torch.load('/users/jrs596/scratch/dat/models/efficientnet-IR.pth')

labels = ["Original", "DisNet_pico", "ResNet", "EfficientNetV2"]

models = [DisNet_pico, DisNet_pico, resnet, efficientnet]
dat_dir = '/users/jrs596/scratch/dat/IR_RGB_Comp_data/IR_split_400/val'
n_imgs = 28

# Prepare data
#img_size = 494

def load_data(dat_dir, img_size):
    data_transforms = transforms.Compose([
        #transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])

    valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    return valloader




# Iterate over models
model_set = [(DisNet_pico, DisNet_pico_CAM, 400)]#, (DisNet_pico, DisNet_pico_CAM, 494), (resnet, ResNet_CAM, 494), (efficientnet, EfficientNet_CAM, 494)]
# Prepare tensor to store images
n_models = len(model_set) + 1  # Add one for the original images
imgs = torch.Tensor(n_imgs, n_models, 3, 400, 400)  # Adjusted the tensor dimensions

for j, model_set in enumerate(model_set, start=1):  # Start from 1 to leave space for the original images
    print() 
    print(labels[j])
    print()
    img_size = model_set[2]
    
    valloader = load_data(dat_dir, img_size)
    it = iter(valloader)
    # Iterate over images
    for i in range(n_imgs):
        img, _ = next(it)
        if j == 1:  # Store the original image only once
            imgs[i][0] = img[0]
            
        model_cam_net = model_set[1](model_set[0], -1)  # Use the final layer
        imgs[i][j] = get_grad_cam(model_cam_net, img)  # Adjusted the tensor indexing


# Transpose the tensor
imgs = imgs.permute(0, 1, 2, 3, 4)  # Adjusted the permute order

# Convert tensor to numpy array
imgs_np = imgs.numpy()

#%%

# Create a figure and axes
fig, axs = plt.subplots(n_imgs, n_models, figsize=(100, 100))
#set plot width
fig.set_figwidth(40)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

class_labels = ["BPR", "FPR", "Healthy", "WBD"]  # Replace with your actual class labels

from PIL import Image, ImageDraw, ImageFont


font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 40)  # Replace with the path to a .ttf file on your system

# Create a new blank image
grid_img = Image.new('RGB', (n_models * img_size, n_imgs * img_size))
# Prepare a list to store the ground truth labels
ground_truth_labels = []

# Iterate over images
it = iter(valloader)
for i in range(n_imgs):
    img, label = next(it)
    #imgs[0][i] = img[0]  # Store the original image
    ground_truth_labels.append(label)  # Store the ground truth label

    # The rest of your code...

# Create a new PIL Image for each model and image
it = iter(valloader)
for i in range(n_imgs):
    # img, label = next(it)
    # imgs[0][i] = img[0]  # Store the original image
    # ground_truth_labels.append(label)  # Store the ground truth label
    for j in range(n_models):
        # Get the image
        img = imgs_np[i, j]
        
        # Convert image from PyTorch format (C, H, W) to matplotlib format (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        
        # Normalize image to [0, 1] range
        img = (img - img.min()) / (img.max() - img.min())
        
        # Convert to PIL Image
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # Create a draw object
        draw = ImageDraw.Draw(img_pil)
        
        # Add class or class prediction to each image
        if j == 0:  # For the first column, use the ground truth label
            label = ground_truth_labels[i]
            draw.text((0, 0), class_labels[label], fill="white", font=font)
        else:  # For all other columns, use the predicted class
            #img to tensor
            img = torch.Tensor(img).permute(2,0,1).to('cuda')
            pred = models[j-1](img.unsqueeze(0))
            pred = pred.argmax(dim=1)
            draw.text((0, 0), class_labels[pred], fill="white", font=font)
        
        # Paste the image onto the grid image
        grid_img.paste(img_pil, (j * img_size, i * img_size))

# Save the grid image
grid_img.save("/users/jrs596/scratch/dat/gradcam_all_models_labeled.png")










# %%
