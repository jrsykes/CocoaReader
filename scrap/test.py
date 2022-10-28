import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
from sklearn import metrics
import pandas as pd
import time
from matplotlib import pyplot as plt 
import time


root = '/local/scratch/jrs596/dat/tmp/norm'
data_dir = "/local/scratch/jrs596/dat/split_cocoa_images"






input_size = 750
batch_size = 37

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
	for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
	shuffle=False, num_workers=1) for x in ['train', 'val']}#


name = 0
for inputs, labels in dataloaders_dict['train']:
	if name < 100:
		name += 1
		flat_image = torch.flatten(inputs).numpy()
		plt.clf()
		plt.hist(flat_image)
		plt.savefig(os.path.join(root, str(name)))
	else:
		exit() 
				




