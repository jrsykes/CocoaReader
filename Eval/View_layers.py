#%%

import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
import toolbox
from ArchitectureZoo import DisNet_nano, DisNet_pico, DisNet_pico_0_1


model = torch.load('/local/scratch/jrs596/dat/models/DisNet-Pico.pth')
model.eval()

print(model)


#%%
# Load the image
image_path = '/local/scratch/jrs596/dat/IR_RGB_Comp_data/RGB_split_400/val/BPR/F4140024.JPG'  # replace with your image path
image = Image.open(image_path)

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to tensor
])

# Apply the transformations to the image
image = transform(image)


# Add an extra dimension to the image tensor and move it to the device
image = image.unsqueeze(0).to("cuda")

# Pass the image through the model up to the first convolutional block
with torch.no_grad():
    x = image
    #x = model.color_grading(x)
    x = model.conv1(x)
    conv1_output = model.cnblock1(x)
    x = model.pool(conv1_output)
    x = model.conv2(x)
    conv2_output = model.cnblock2(x)

# Detach the outputs from the computation graph and move them to the CPU
conv1_output = conv1_output.detach().cpu()
conv2_output = conv2_output.detach().cpu()

# Convert the tensors to numpy arrays
conv1_output = conv1_output.numpy()
conv2_output = conv2_output.numpy()

# Plot the output of the first convolutional block
plt.figure(figsize=(10, 10))
for i in range(conv1_output.shape[1]):
    plt.subplot(4, 4, i+1)  # Assuming the output has 64 channels
    plt.imshow(conv1_output[0, i, :, :], cmap='gray')
    plt.axis('off')
plt.show()

# Plot the output of the second convolutional block
plt.figure(figsize=(10, 10))
for i in range(conv2_output.shape[1]):
    plt.subplot(7, 8, i+1)  # Assuming the output has 64 channels
    plt.imshow(conv2_output[0, i, :, :], cmap='gray')
    plt.axis('off')
plt.show()

#%
# %%
