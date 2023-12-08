#%%
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import torch.nn.functional as F
import os
from torchvision import datasets, transforms, models
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
from torchvision.transforms import ToPILImage


device = torch.device("cuda:0")

#%%
   
class ResNet_CAM(nn.Module):
    def __init__(self, net, layer_k=-1):
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
    img_as_float = img[0].permute(1,2,0).cpu().numpy().astype(float)

    superimposed_img = cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB) * 0.006 + img_as_float
    superimposed_img = torch.from_numpy(superimposed_img)  # Convert back to tensor if necessary

    return superimposed_img



def get_grad_cam(net, img):
    net.eval()
    img = img.to(device)
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

    
    return superimpose_heatmap(heatmap, img)



resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-IN-PT.pth"
ResNet18Weights = torch.load(resnet18_cococa_weights, map_location=device)

ResNet18 = models.resnet18(weights=None)
in_feat = ResNet18.fc.in_features
ResNet18.fc = nn.Linear(in_feat, 4)
ResNet18.load_state_dict(ResNet18Weights, strict=True)
ResNet18.to(device)

dat_dir = '/users/jrs596/scratch/dat/Ecuador/GradCAM_TestImgs/'



def load_data(dat_dir, img_size):
    data_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])

    valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    return valloader



img_size = 375
plot_img_size = img_size
valloader = load_data(dat_dir, img_size)
it = iter(valloader)

n_imgs = len(valloader)
n_models = 2
imgs = torch.Tensor(n_imgs, n_models, 3, plot_img_size, plot_img_size)  # Adjusted the tensor dimensions

for i in range(n_imgs):
    img, _ = next(it)

    imgs[i][0] = img
    
    model_cam_net = ResNet_CAM(ResNet18)
    
    out = get_grad_cam(model_cam_net, img)  
    #Normalise GradCAM output
    out = (out - out.min()) / (out.max() - out.min())
    imgs[i][1] = out.permute(2,0,1).cpu().mul(255).byte()

    # ###############################
    #convert to PIL image
    im1 = img.squeeze().permute(1,2,0).cpu().numpy()
    im1 = Image.fromarray((im1 * 255).astype(np.uint8))
    
    im2 = out.permute(2,0,1).cpu().mul(255).byte()
    im2 = ToPILImage()(im2)

    combined_width = im1.width + im2.width
    combined_height = max(im1.height, im2.height)

    # Create a new blank image with the combined dimensions
    new_img = Image.new('RGB', (combined_width, combined_height))

    # Paste the images side by side
    new_img.paste(im1, (0, 0))
    new_img.paste(im2, (img_size, 0))

    # Save the image
    new_img.save("/users/jrs596/scratch/dat/Ecuador/GradCAM_ResNet18_IN-PT/" + str(i) + ".png")
    


# Convert tensor to numpy array
imgs_np = imgs.numpy()

#%%
print("\nMapping complete, getting predictions.\n")
# Create a figure and axes
fig, axs = plt.subplots(n_imgs, n_models, figsize=(100, 100))
#set plot width
fig.set_figwidth(40)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

class_labels = ["BPR", "FPR", "Healthy", "WBD"]  # Replace with your actual class labels

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 80)  # Replace with the path to a .ttf file on your system

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
        
        # Add class or class prediction to each image
        if j == 0:  # For the first column, use the ground truth label
            label = ground_truth_labels[i]
            draw.text((0, 0), class_labels[label], fill="white", font=font)
        else:  # For all other columns, use the predicted class
            # resize img_ tensor to 3 x 224 x 224
            img__ = torch.nn.functional.interpolate(img_, size=img_size, mode='bilinear', align_corners=False)
            model = ResNet18
            model.to(device)
            model.eval()

            pred = model(img__.to(device))
            pred = pred.argmax(dim=1)
            draw.text((0, 0), class_labels[pred], fill="white", font=font)
        
        # out = img_pil, (j * img_size, i * img_size)

        # Paste the image onto the grid image
        grid_img.paste(img_pil, (j * img_size, i * img_size))

# Save the grid image
grid_img.save("/users/jrs596/scratch/dat/Ecuador/GradCAM_ResNet18_IN-PT.png")



print("Done!")






# %%
