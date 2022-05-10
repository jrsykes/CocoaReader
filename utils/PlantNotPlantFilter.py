import pandas as pd
from matplotlib import pyplot as plt
import os
import statistics
from math import sqrt
from scipy.stats import norm
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import shutil

#from __future__ import print_function
#from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
from sklearn import metrics
from progress.bar import Bar


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean_unorganised/"
#data_dir = '/local/scratch/jrs596/dat/test3'
data_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean/train/'

input_size = 64
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# File name for model
model_name = "PlantNotPlant_IMTiny"

# Number of classes in the dataset
num_classes = 2
###############################################
transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)


val_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)



#Initialize and Reshape the Networks
def initialize_model(num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None

    model = models.resnet18(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

# Initialize the model for this run
model = initialize_model(num_classes, use_pretrained=True)

pretrained_model_path = os.path.join('/local/scratch/jrs596/ResNetFung50_Torch/models', model_name + '.pkl')
pretrained_model_wts = pickle.load(open(pretrained_model_path, "rb"))

weights = copy.deepcopy(pretrained_model_wts['model'])

###############
#Remove 'module.' from layer names
new_keys = []
for key, value in weights.items():
    new_keys.append(key.replace('module.', ''))
for i in new_keys:
    weights[i] = weights.pop('module.' + i)#    

##############

model.load_state_dict(weights)

model.eval()
model.to(device)


##############################################

index = 0


for inputs, labels in loader:
	class_ = dataset.classes[int(labels[0])]
	source = val_dataset.imgs[index][0]
	inputs = inputs.to(device)
	outputs = model(inputs)
	_, preds = torch.max(outputs, 1)
	
	os.makedirs(os.path.join('/local/scratch/jrs596/dat/PlantNotPlant_TinyIM_Filtered/', class_), exist_ok=True)
	if preds.item() == 1:
		dest = os.path.join('/local/scratch/jrs596/dat/PlantNotPlant_TinyIM_Filtered/', class_, str(index) + '.jpg')
		shutil.copy(source,dest)
		
	print(source)
	index += 1
