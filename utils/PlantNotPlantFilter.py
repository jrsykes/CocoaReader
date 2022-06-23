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
from PIL import Image
from difPy import dif


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_clean"
#data_dir = '/local/scratch/jrs596/dat/ILSVRC/Data/CLS-LOC'

input_size = 224
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda")   # use CPU or GPU

# File name for model
model_name = "PlantNotPlant_final_unorganised"

# Number of classes in the dataset
num_classes = 2
###############################################
transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



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
# Filter out non .JPEG and corrupt image files
index = 2363
#for class_ in os.listdir(data_dir):
#	for image in os.listdir(os.path.join(data_dir, class_)):
#		file_path = os.path.join(data_dir,class_,image)
#		try:
#			im = Image.open(file_path, formats=['JPEG'])
#		except:
#			dest = os.path.join('/local/scratch/jrs596/dat/final_filter/corrupt', str(index) + '.jpg')
#			shutil.move(file_path,dest)
#			index += 1
#			print(file_path)#

###############################################
## Delete duplicate images for each class
#for i in os.listdir(data_dir):
#	search = dif(os.path.join(data_dir, i), delete=True, silent_del=True)


##############################################
# Filter out non-plant images with Plant-NotPlant CNN
dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

for i, (inputs, labels) in enumerate(loader, 0):
	source, _ = loader.dataset.samples[i]
	inputs = inputs.to(device)
	outputs = model(inputs)
	# [0][1] = Plant
	# [0][0] = NotPlant
	if torch.sigmoid(outputs)[0][0].item() > 0.995:
		dest = os.path.join('/local/scratch/jrs596/dat/final_filter/PNP', str(index) + '.jpg')
		shutil.move(source,dest)
		print()
		print(source)
		print(torch.sigmoid(outputs)[0])

		index += 1