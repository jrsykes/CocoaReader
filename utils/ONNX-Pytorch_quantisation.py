import sys
from torchvision import models, datasets, transforms as T
import torch
import os
import pickle

root = '/local/scratch/jrs596/dat/models'
model_path = 'CocoaNet18_DN.pth'
model_weights = 'CocoaNet18_DN.pkl'
data_dir = "/local/scratch/jrs596/dat/split_cocoa_images"


CocoaNet = torch.load(os.path.join(root, model_path), map_location=torch.device('cpu'))



image_height = 224
image_width = 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)


torch_out = CocoaNet(x)

print(torch_out.size())