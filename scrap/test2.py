
#%%
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
#import models
#from torchbearer import Trial
#%%
#import cv2
#import torch.nn.functional as F
#import os
from torchvision import datasets, transforms
import sys
sys.path.append('/users/jrs596/scripts/CocoaReader/utils')
#from ArchitectureZoo import DisNet_pico
import torch
#import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
#from ArchitectureZoo import DisNet_pico, DisNet_pico_deep, DisNet_nano

#load an image
img_path = "/users/jrs596/scratch/dat/IR_RGB_Comp_data/IR_split_400/val/BPR/F4130164.JPG"

#load image
img = Image.open(img_path)


class AveragePoolingColorChannels(object):
    def __call__(self, img):
        return torch.mean(img, dim=0)

# Define your transform
transform = transforms.Compose([
    transforms.ToTensor(),
    AveragePoolingColorChannels(),
])

# Load your image (replace 'image.jpg' with your image path)
from PIL import Image
image = Image.open('image.jpg')

gray_image = transform(image)

#display gray_image tensor as greyscale image
plt.imshow(gray_image.squeeze(), cmap='gray')
#%%