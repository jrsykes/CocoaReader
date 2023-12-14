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
# device = torch.device("cpu")

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


out_labels = ["BPR", "FPR", "Healthy", "WBD"]  

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 80)  # Replace with the path to a .ttf file on your system


FullSup_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa_FullSup-IN-PT.pth"
SemiSup_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-IN-PT.pth"

models_ = ["Input", FullSup_weights, SemiSup_weights]

def load_model(weights_pth):
    ResNet18Weights = torch.load(weights_pth, map_location=device)
    ResNet18 = models.resnet18(weights=None)
    in_feat = ResNet18.fc.in_features
    ResNet18.fc = nn.Linear(in_feat, 4)
    ResNet18.load_state_dict(ResNet18Weights, strict=True)
    ResNet18.eval()
    ResNet18.to(device)
    
    return ResNet18

dat_dir = '/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Easy/val/'
class_labels = os.listdir(dat_dir)
class_labels.sort()

def load_data(dat_dir, img_size):
    data_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])

    valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=1)

    return valloader

#%%

img_size = 375
n_imgs = 40
n_models = 2

plot_img_size = img_size
valloader = load_data(dat_dir, img_size)


imgs = torch.Tensor(n_imgs, n_models, 3, plot_img_size, plot_img_size)  # Adjusted the tensor dimensions
imgs_np = imgs.numpy()
ground_truth_labels = []

fig, axs = plt.subplots(n_imgs, n_models, figsize=(100, 100))
fig.set_figwidth(40)
plt.subplots_adjust(wspace=0, hspace=0)
# Create a new blank image
grid_img = Image.new('RGB', (n_models * img_size, n_imgs * img_size))


    
#%%    


for i, (img, label) in enumerate(valloader):
    if i >= n_imgs:
        break


    
    for idx, model in enumerate(models_):
        if idx == 0:  # For the first column, use the ground truth label
            pil_img = Image.fromarray((img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            draw = ImageDraw.Draw(pil_img)
            draw.text((0, 0), class_labels[label], fill="white", font=font)
            grid_img.paste(pil_img, (idx * img_size, i * img_size))
        elif idx == 1:  # For the second column, use the Grad-CAM output with predicted class
            model = load_model(model)
            model_cam_net = ResNet_CAM(model)

            out = get_grad_cam(model_cam_net, img)  
            #Normalise GradCAM output
            out = (out - out.min()) / (out.max() - out.min())
            out = out.permute(2,0,1).cpu().mul(255).byte()

            pred = model(img.to(device))
            pred = pred.argmax(dim=1)

            # Convert out to PIL image and draw the predicted class
            out_pil_1 = ToPILImage()(out)
            draw = ImageDraw.Draw(out_pil_1)
            draw.text((0, 0), out_labels[pred], fill="white", font=font)

            grid_img.paste(out_pil_1, (idx * img_size, i * img_size))
        elif idx == 2:  # For the second column, use the Grad-CAM output with predicted class
            model = load_model(model)
            model_cam_net = ResNet_CAM(model)

            out = get_grad_cam(model_cam_net, img)  
            #Normalise GradCAM output
            out = (out - out.min()) / (out.max() - out.min())
            out = out.permute(2,0,1).cpu().mul(255).byte()

            pred = model(img.to(device))
            pred = pred.argmax(dim=1)

            # Convert out to PIL image and draw the predicted class
            out_pil_2 = ToPILImage()(out)
            draw = ImageDraw.Draw(out_pil_2)
            draw.text((0, 0), out_labels[pred], fill="white", font=font)

            grid_img.paste(out_pil_2, (idx * img_size, i * img_size))
            
    # Create a new blank image with the combined dimensions
    new_img = Image.new('RGB', (img_size*len(models_), img_size))
    # Paste the images side by side
    new_img.paste(pil_img, (0, 0))
    new_img.paste(out_pil_1, (img_size, 0))
    new_img.paste(out_pil_2, (img_size*2, 0))

    # Save the image
    dir_ = "/users/jrs596/GradCAM_imgs/GradCAM_ResNet18_IN-PT_Full-SemiSup"
    os.makedirs(dir_, exist_ok=True)
    new_img.save(os.path.join(dir_, str(i) + ".png"))

grid_img.save("/users/jrs596/GradCAM_imgs/GradCAM_ResNet18_IN-PT_Full-SemiSup.png")

print("Done!")

# for i, (img, label) in enumerate(valloader):
#     if i >= n_imgs:
#         break

#     out = get_grad_cam(model_cam_net, img)  
#     #Normalise GradCAM output
#     out = (out - out.min()) / (out.max() - out.min())
#     out = out.permute(2,0,1).cpu().mul(255).byte()
    
#     for j in range(n_models):
#         if j == 0:  # For the first column, use the ground truth label
#             pil_img = Image.fromarray((img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
#             draw = ImageDraw.Draw(pil_img)
#             draw.text((0, 0), class_labels[label], fill="white", font=font)
#             grid_img.paste(pil_img, (j * img_size, i * img_size))
#         elif j == 1:  # For the second column, use the Grad-CAM output with predicted class
#             model = ResNet18
#             model.to(device)
#             model.eval()

#             pred = model(img.to(device))
#             pred = pred.argmax(dim=1)

#             # Convert out to PIL image and draw the predicted class
#             out_pil = ToPILImage()(out)
#             draw = ImageDraw.Draw(out_pil)
#             draw.text((0, 0), out_labels[pred], fill="white", font=font)

#             grid_img.paste(out_pil, (j * img_size, i * img_size))
            
#     # Create a new blank image with the combined dimensions
#     new_img = Image.new('RGB', (img_size*2, img_size))
#     # Paste the images side by side
#     new_img.paste(pil_img, (0, 0))
#     new_img.paste(out_pil, (img_size, 0))
#     # Save the image
#     dir_ = "/users/jrs596/GradCAM_imgs/GradCAM_ResNet18_IN-PT_FullSup"
#     os.makedirs(dir_, exist_ok=True)
#     new_img.save(os.path.join(dir_, str(i) + ".png"))

# grid_img.save("/users/jrs596/GradCAM_imgs/GradCAM_ResNet18_IN-PT_FullSup.png")

# print("Done!")


