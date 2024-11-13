import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
from sklearn import metrics
import pandas as pd
import time
import numpy as np
#import cv2
import sys

sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')

import toolbox



data_dir = "/local/scratch/jrs596/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca/Easy"
num_classes = len(os.listdir(os.path.join(data_dir, 'val'))) 

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

resnet18_cococa_weights = "/users/jrs596/scratch/models/CocoaNet18_V1.1.pth"

ResNet18Weights = torch.load(resnet18_cococa_weights, map_location=device)

model = models.resnet18(weights=None)
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, 5)
model.load_state_dict(ResNet18Weights, strict=True)



model.eval()   # Set model to evaluate mode
model_RGB = model.to(device)
model_MS = model.to(device)
model_Pol = model.to(device)

batch_size = 2

input_size = 375
image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=input_size) #If images are pre compressed, use input_size=None, else use input_size=args.input_size
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=10, worker_init_fn=toolbox.worker_init_fn, drop_last=True) for x in ['train', 'val']}


N = len(dataloaders_dict['val'])
FPS = 0
   
start = time.time()

for idx, (inputs, labels) in enumerate(dataloaders_dict['val']):
	inputs = inputs.to(device)
	outputs = model_RGB(inputs)
	outputs = model_MS(inputs)
	outputs = model_Pol(inputs)

# FPS calculation
FPS = (N * batch_size) / (time.time() - start)
print("FPS: ", FPS)
