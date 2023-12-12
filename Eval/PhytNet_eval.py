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

sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')

data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_SplitCompress500_2/Easy"

device = torch.device("cuda:0")

config = {
        'beta1': 0.9650025364732508,
        'beta2': 0.981605256508036,
        'dim_1': 79,
        'dim_2': 107,
        'dim_3': 93,
        'input_size': 415,
        'kernel_1': 5,
        'kernel_2': 1,
        'kernel_3': 7,
        'learning_rate': 0.0002975957026209971,
        'num_blocks_1': 2,
        'num_blocks_2': 1,
        'out_channels': 6
    }
	
model = toolbox.build_model(num_classes=None, arch='PhytNetV0', config=config)

weights_path = "/users/jrs596/scratch/models/PhytNet-Cocoa-SR-PT.pth"

PhyloNetWeights = torch.load(weights_path, map_location=device)


model.load_state_dict(PhyloNetWeights, strict=True)
input_size = 415
print('\nLoaded weights from: ', weights_path)

# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-IN-PT.pth"
# ResNet18Weights = torch.load(resnet18_cococa_weights, map_location=device)

# model = models.resnet18(weights=None)
# in_feat = model.fc.in_features
# model.fc = nn.Linear(in_feat, 4)
# model.load_state_dict(ResNet18Weights, strict=True)
# input_size = 375

model.eval()   # Set model to evaluate mode
model = model.to(device)

batch_size = 1
criterion = nn.CrossEntropyLoss()

image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=input_size) #If images are pre compressed, use input_size=None, else use input_size=args.input_size
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=6, worker_init_fn=toolbox.worker_init_fn, drop_last=False) for x in ['train', 'val']}

my_metrics = toolbox.Metrics(metric_names='All', num_classes=4)

N = len(dataloaders_dict['val']) + len(dataloaders_dict['train'])
start = time.time()
for phase in ['train', 'val']:
	for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
		inputs = inputs.to(device)
		labels = labels.to(device)
		
		_, _, outputs = model(inputs)

		loss = criterion(outputs, labels)

		_, preds = torch.max(outputs, 1)    
		stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
		stats_out = stats['weighted avg']
					   
		my_metrics.update(loss=loss, preds=preds, labels=labels, stats_out=stats_out)

	epoch_metrics = my_metrics.calculate()
	
	print()
	print(phase)
	print(epoch_metrics)
	my_metrics.reset()


print("Frames per second: ", N /(time.time() - start))