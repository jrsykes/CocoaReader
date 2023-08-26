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
sys.path.append('/users/jrs596/scripts/CocoaReader/utils')
from ArchitectureZoo import DisNet_pico, DisNet_picoIR
import toolbox 

device = torch.device("cuda:0")

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


class EfficientNet_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(EfficientNet_CAM, self).__init__()
        self.efficientnet = net
        convs = nn.Sequential(*list(net.children())[:-1])
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:-1]  # Exclude the last layer
        self.linear = nn.Sequential(convs[-1], *list(net.children())[-1:])  # Include the last layer in linear

        
    def forward(self, x):
        x = self.first_part_conv(x.to(device))
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
        return self.first_part_conv(x.to(device))


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
# Load models
config = {
            'num_classes': 8,
            "dim_1": 20,
            "dim_2": 51,
            "drop_out": 0.3525889518248164,
            "drop_out2": 0.2560665455502205,
            "input_size": 240,
            "kernel_1": 7,
            "kernel_2": 11,
            "kernel_3": 12,
            "kernel_4": 3,
            "kernel_5": 5,
            "kernel_6": 2,
            "nodes_1": 71,
            "nodes_2": 144,
            "nodes_3": 91,
            "nodes_4": 132,
            "trans_nodes": 108
            }

  
DisNet = toolbox.build_model(num_classes=config['trans_nodes'], arch='DisNet_picoIR', config=config)
SecondNet = toolbox.build_model(num_classes=config['trans_nodes'], arch='resnet18', config=None)
Meta = toolbox.build_model(num_classes=config['num_classes'], arch='Meta', config=config)

config2 = {'CNN1': DisNet, 
           'CNN2': SecondNet, 
           'MetaModel': Meta}

DisResNet = toolbox.build_model(num_classes=None, arch='Unified', config=config2).to(device)  

# DisNet_pico = DisNet_pico(out_channels=4).to(device)

# # Load weights
weights_path = "/users/jrs596/scratch/dat/models/DisNet-IR-lemon-sweep-620_weights.pth"
weights = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(0))

def remove_total_weights(model_state_dict):
    """
    Remove weights with "total" in the name from a PyTorch model's state dictionary.

    Parameters:
        model_state_dict (OrderedDict): The state dictionary of a PyTorch model.

    Returns:
        new_state_dict (OrderedDict): The modified state dictionary.
    """
    new_state_dict = {}
    for key, value in model_state_dict.items():
        if "total" not in key:
            new_state_dict[key] = value
    return new_state_dict

state_dict = remove_total_weights(weights)

# # Apply weights to the model
DisResNet.load_state_dict(state_dict, strict=False)
#%%


DisNet_pico = DisResNet.cnn1
resnet = DisResNet.cnn2

#%%


labels = ["Original", "DisNet_pico", "ResNet"]
# labels = ["Original", "ResNet", "Efficientnet_b0", "EfficientNetV2"]

models = [DisNet_pico, resnet]
# models = [resnet, efficientnet, efficientnet_b0]

dat_dir = '/users/jrs596/scratch/dat/IR_RGB_Comp_data/GradCamTest40'
n_imgs = 40

# Prepare data
# img_size = config['input_size']

def load_data(dat_dir, img_size):
    data_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])

    valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    return valloader




# Iterate over models
model_list = [(DisNet_pico, DisNet_pico_CAM, 240), (resnet, ResNet_CAM, 408)]
plot_img_size = 408
# Prepare tensor to store images
n_models = len(model_list) + 1  # Add one for the original images
imgs = torch.Tensor(n_imgs, n_models, 3, plot_img_size, plot_img_size)  # Adjusted the tensor dimensions

for j, model_set in enumerate(model_list, start=1):  # Start from 1 to leave space for the original images
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
            img_resize = torch.nn.functional.interpolate(img[0].unsqueeze(0), size=(plot_img_size), mode='bilinear', align_corners=False)
            imgs[i][0] = img_resize
            # imgs[i][0] = img[0]
            
        model_cam_net = model_set[1](model_set[0], -1)  # Use the final layer
        out = get_grad_cam(model_cam_net, img)
        # resize out tensor to 3 x 224 x 224
        out = torch.nn.functional.interpolate(out.unsqueeze(0), size=(plot_img_size), mode='bilinear', align_corners=False)
        imgs[i][j] = out  # Adjusted the tensor indexing


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



font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 40)  # Replace with the path to a .ttf file on your system

# Create a new blank image
grid_img = Image.new('RGB', (n_models * img_size, n_imgs * img_size))
# Prepare a list to store the ground truth labels
ground_truth_labels = []

valloader = load_data(dat_dir, 1000)

# Create a new PIL Image for each model and image
it = iter(valloader)
for i in range(n_imgs):
    img_, label = next(it)
    ground_truth_labels.append(label)  # Store the ground truth label
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
        
        # # Add class or class prediction to each image
        # if j == 0:  # For the first column, use the ground truth label
        #     label = ground_truth_labels[i]
        #     draw.text((0, 0), class_labels[label], fill="white", font=font)
        # else:  # For all other columns, use the predicted class
        #     # resize img_ tensor to 3 x 224 x 224
        #     img__ = torch.nn.functional.interpolate(img_, size=model_list[j-1][2], mode='bilinear', align_corners=False)
        #     try:
        #         pred, _ = models[j-1](img__.to(device))
        #     except:
        #         pred = models[j-1](img__.to(device))
        #     pred = pred.argmax(dim=1)
        #     draw.text((0, 0), class_labels[pred], fill="white", font=font)
        
        # Paste the image onto the grid image
        grid_img.paste(img_pil, (j * img_size, i * img_size))

# Save the grid image
grid_img.save("/users/jrs596/scratch/dat/gradcam_DisResNet.png")
# grid_img.save("/users/jrs596/scratch/dat/gradcam_DisNet-duo.png")



print("Done!")






# %%
