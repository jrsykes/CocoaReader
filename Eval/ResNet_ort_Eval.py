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

input_size = 375


########################################

# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-SemiSupervised_NotCocoa_DFLoss2.pth"

# ResNet18Weights = torch.load(resnet18_cococa_weights, map_location=device)

# model = models.resnet18(weights=None)
# in_feat = model.fc.in_features
# model.fc = nn.Linear(in_feat, 5)
# model.load_state_dict(ResNet18Weights, strict=True)

import onnxruntime as ort

# Load the ONNX model
# session = ort.InferenceSession("/local/scratch/jrs596/models/CocoaNet18_2024-03-02.ort")

session = ort.InferenceSession("/local/scratch/jrs596/models/CocoaNet18_2024-03-02.ort", providers=['CUDAExecutionProvider'])

# Get the name of the input node
input_name = session.get_inputs()[0].name

batch_size = 6

image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=input_size) #If images are pre compressed, use input_size=None, else use input_size=args.input_size
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=10, worker_init_fn=toolbox.worker_init_fn, drop_last=True) for x in ['train', 'val']}


N = len(dataloaders_dict['val'])
FPS = 0

start = time.time()
# for phase in ['train', 'val']:
for inputs, labels in dataloaders_dict['val']:
    # Process each input in the batch individually
    for i in range(inputs.shape[0]):  # Loop over each example in the batch
        # Convert a single PyTorch tensor to NumPy array and ensure it's on the CPU
        # Use [i:i+1] to keep the batch dimension as 1
        input_np = inputs[i:i+1].cpu().detach().numpy()
        
        # Run inference
        output = session.run(None, {input_name: input_np})
        # print(output)

# FPS calculation
FPS = (N * batch_size) / (time.time() - start)
Batch_per_second = FPS / batch_size
print("FPS: ", FPS)
print("Batch per second: ", Batch_per_second)

##################################

