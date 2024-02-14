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
import toolbox

import subprocess
import re

sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')

# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Unsure"
# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Difficult"

# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca/Easy"
# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Easy"


data_dir = "/users/jrs596/scratch/dat/IR_split"
num_classes = len(os.listdir(os.path.join(data_dir, 'val'))) 

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#67k - effortless-sweep-30
# config = {
#         'beta1': 0.9650025364732508,
#         'beta2': 0.981605256508036,
#         'dim_1': 79,
#         'dim_2': 107,
#         'dim_3': 93,
#         'input_size': 415,
#         'kernel_1': 5,
#         'kernel_2': 1,
#         'kernel_3': 7,
#         'learning_rate': 0.0002975957026209971,
#         'num_blocks_1': 2,
#         'num_blocks_2': 1,
#         'out_channels': 6
#     }

#smart-sweep-47 332k
# config = {
#     'beta1': 0.9657828624377116,
#     'beta2': 0.9908102731106424,
#     'dim_1': 104,
#     'dim_2': 109,
#     'dim_3': 110,
#     'input_size': 350,
#     'kernel_1': 5,
#     'kernel_2': 7,
#     'kernel_3': 13,
#     'learning_rate': 0.00013365304940966892,
#     'num_blocks_1': 1,
#     'num_blocks_2': 2,
#     'out_channels': 9
# }

# cool-sweep-42
# config = {
#     'beta1': 0.9671538235629524,
#     'beta2': 0.9574398373980104,
#     'dim_1': 126,
#     'dim_2': 91,
#     'dim_3': 89,
#     'input_size': 371,
#     'kernel_1': 5,
#     'kernel_2': 1,
#     'kernel_3': 17,
#     'learning_rate': 9.66816458944127e-05,
#     'num_blocks_1': 2,
#     'num_blocks_2': 1,
#     'out_channels': 7
# }
	
config = {
        "beta1": 0.9051880132274126,
        "beta2": 0.9630258300974864,
        "dim_1": 49,
        "dim_2": 97,
        "dim_3": 68,
        "kernel_1": 11,
        "kernel_2": 9,
        "kernel_3": 13,
        "learning_rate": 0.0005921981578304907,
        "num_blocks_1": 2,
        "num_blocks_2": 4,
        "out_channels": 7,
        "input_size": 285,
    }
 
model = toolbox.build_model(num_classes=config['out_channels'], arch='PhytNetV0_ablation', config=config)

# weights_path = "/users/jrs596/scratch/models/PhytNet183k-Cocoa-SemiSupervised_NotCocoa_DFLoss2.pth"
# weights_path = "/users/jrs596/scratch/models/PhytNet67k-Cocoa-SemiSupervised_NotCocoa_OptDFLoss.pth"

weights_path = '/users/jrs596/scratch/models/PhytNet-Cocoa-ablation.pth'

PhyloNetWeights = torch.load(weights_path, map_location=device)


model.load_state_dict(PhyloNetWeights, strict=True)
input_size = config['input_size']
print('\nLoaded weights from: ', weights_path)

# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-SemiSupervised_NotCocoa.pth"
# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-IN-PT.pth"
# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-SemiSupervised_NotCocoa_DFLoss2.pth"

# ResNet18Weights = torch.load(resnet18_cococa_weights, map_location=device)

# model = models.resnet18(weights=None)
# in_feat = model.fc.in_features
# model.fc = nn.Linear(in_feat, 5)
# model.load_state_dict(ResNet18Weights, strict=True)
# input_size = 375

model.eval()   # Set model to evaluate mode
model = model.to(device)

batch_size = 42
criterion = nn.CrossEntropyLoss()

image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=input_size) #If images are pre compressed, use input_size=None, else use input_size=args.input_size
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=10, worker_init_fn=toolbox.worker_init_fn, drop_last=True) for x in ['train', 'val']}

my_metrics = toolbox.Metrics(metric_names=['loss', 'corrects', 'precision', 'recall', 'f1'], num_classes=num_classes)

N = len(dataloaders_dict['val']) + len(dataloaders_dict['train'])

FPS = 0

input_size = torch.Size([3, input_size, input_size])
   
# GFLOPs, n_params = toolbox.count_flops(model=model, device=device, input_size=input_size)
# print("GFLOPS:", GFLOPs)

# print(N)
# exit()
# for i in range(10):
start = time.time()
for phase in ['train', 'val']:
	for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
		inputs = inputs.to(device)
		labels = labels.to(device)
		_,_,outputs = model(inputs)
		# outputs = model(inputs)
	loss = criterion(outputs, labels)
	_, preds = torch.max(outputs, 1)    
	stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), output_dict = True)
	stats_out = stats['weighted avg']
				   
	my_metrics.update(loss=loss, preds=preds, labels=labels, stats_out=stats_out)
	epoch_metrics = my_metrics.calculate()
	print()
	print(phase)
	print(epoch_metrics)
	my_metrics.reset()

# FPS calculation
FPS = (N * batch_size) / (time.time() - start)
print("FPS: ", FPS)

##################################


# def get_gpu_utilization():
#     try:
#         nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).decode("utf-8")
#         gpu_utilization = int(re.findall(r'\d+', nvidia_smi_output)[0])
#         return gpu_utilization
#     except Exception as e:
#         print(f"Error fetching GPU utilization: {e}")
#         return None

# # Assuming you have a model, dataloaders_dict, and device defined
# # Start time for FPS calculation
# start = time.time()

# # Initial GPU memory usage
# initial_memory = torch.cuda.memory_allocated(device)
# peak_memory = initial_memory

# # Evaluation loop
# gpu_utilizations = []
# for phase in ['train', 'val']:
#     for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
#         inputs = inputs.to(device)

#         with torch.no_grad():  # Ensure no gradients are calculated
#             outputs = model(inputs)

#         # Check and update peak GPU memory usage
#         current_memory = torch.cuda.memory_allocated(device)
#         peak_memory = max(peak_memory, current_memory)

#         # Measure GPU utilization
#         gpu_util = get_gpu_utilization()
#         if gpu_util is not None:
#             gpu_utilizations.append(gpu_util)

# # FPS calculation
# FPS = (N * batch_size) / (time.time() - start)
# print("FPS: ", FPS)

# # GPU Memory Usage
# used_memory = peak_memory - initial_memory
# # print(f"Initial Memory Usage: {initial_memory / (1024**2)} MB")
# print(f"Peak Memory Usage: {peak_memory / (1024**2)} MB")
# # print(f"Memory Used in Evaluation: {used_memory / (1024**2)} MB")

# # Average GPU Utilization
# if gpu_utilizations:
#     average_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations)
#     print(f"Average GPU Utilization: {average_gpu_utilization}%")
# else:
#     print("GPU Utilization data not available.")
