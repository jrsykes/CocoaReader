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
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
from sklearn import metrics
from progress.bar import Bar
from PIL import Image
import matplotlib.pyplot as plt


# File name for model
model_type = 'resnet'
model_name = "PlantNotPlant3.1"
root = '/scratch/staff/jrs596/dat'
#image_out_dir = os.path.join(root, 'Forestry_ArableImages_GoogleBing_PNP_out')


data_dir = os.path.join(root, "CLS-LOC_filtered/second_pass")
#model_path = os.path.join('/scratch/staff/jrs596/PNP_testing/ResNet18_original2')




# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_clean"
#data_dir = '/scratch/staff/jrs596/dat/PlantNotPlant3/test'
#data_dir = '/scratch/staff/jrs596/dat/CLS-LOC_filtered/first_pass'

input_size = 224
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda")   # use CPU or GPU



# File name for model
#model_name = "PlantNotPlant_2_ConvNext"
#model_name = "PlantNotPlant_2"

# Number of classes in the dataset
num_classes = 2
###############################################
transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#Initialize and Reshape the Networks
def initialize_model(num_classes, type):
	# Initialize these variables which will be set in this if statement. Each of these
	#   variables is model specific.
	model = None
	if model_type == 'resnet':
		model = models.resnet18()
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, num_classes)
	elif model_type == 'convnext':
		model = models.convnext_tiny()
		model.classifier[2].out_features = num_classes
	return model

# Initialize the model for this run

model = initialize_model(num_classes, model_type)

pretrained_model_path = os.path.join('/scratch/staff/jrs596/dat/models', model_name + '.pkl')
pretrained_model_wts = pickle.load(open(pretrained_model_path, "rb"))

weights = copy.deepcopy(pretrained_model_wts['model'])

#Remove 'module.' from layer names
new_keys = []
for key, value in weights.items():
    new_keys.append(key.replace('module.', ''))
for i in new_keys:
    weights[i] = weights.pop('module.' + i)

model.load_state_dict(weights)

model.eval()
model.to(device)

##############################################
# Filter out non-plant images with Plant-NotPlant CNN
if model_type == 'resnet':
	dest = '/scratch/staff/jrs596/PNP_testing/ResNet18_original2'
elif model_type == 'convnext':
	dest = '/scratch/staff/jrs596/PNP_testing/ConvNext_original'


print('Loading data')
dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)
print('Data loaded')

print('Testing')
index = 0
for i, (inputs, labels) in enumerate(loader, 0):
	source, _ = loader.dataset.samples[i]
	inputs = inputs.to(device)
	outputs = model(inputs)
	outputs = torch.sigmoid(outputs)
	# [0][1] = Plant
	# [0][0] = NotPlant
	if outputs[0][0].item() < outputs[0][1].item() and outputs[0][1].item() > 0.5:
		#final_dest = os.path.join(dest, str(index) + '.jpg')
		shutil.move(source,dest)
		print()
		print(source)
		print(outputs[0])	
		index += 1
