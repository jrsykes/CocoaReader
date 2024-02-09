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

# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Unsure"
# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Difficult"

data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca/Easy"
# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Easy"

num_classes = len(os.listdir(os.path.join(data_dir, 'val'))) 

device = torch.device("cuda:0")

# config = {
# 		'beta1': 0.9650025364732508,
# 		'beta2': 0.981605256508036,
# 		'dim_1': 79,
# 		'dim_2': 107,
# 		'dim_3': 93,
# 		'input_size': 415,
# 		'kernel_1': 5,
# 		'kernel_2': 1,
# 		'kernel_3': 7,
# 		'learning_rate': 0.0002975957026209971,
# 		'num_blocks_1': 2,
# 		'num_blocks_2': 1,
# 		'out_channels': 6
# 	}
	
# model = toolbox.build_model(num_classes=None, arch='PhytNetV0', config=config)

# weights_path = "/users/jrs596/scratch/models/PhytNet-Cocoa-SemiSupervised_NotCocoa_SR.pth"

# PhyloNetWeights = torch.load(weights_path, map_location=device)


# model.load_state_dict(PhyloNetWeights, strict=True)
# input_size = 415
# print('\nLoaded weights from: ', weights_path)

# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa_FullSup-IN-PT.pth"
# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-IN-PT.pth"
resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-SemiSupervised_NotCocoa.pth"

ResNet18Weights = torch.load(resnet18_cococa_weights, map_location=device)

model = models.resnet18(weights=None)
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, num_classes)
model.load_state_dict(ResNet18Weights, strict=True)
input_size = 375

model.eval()   # Set model to evaluate mode
model = model.to(device)

criterion = nn.CrossEntropyLoss()

batch_size = 1

image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=input_size) #If images are pre compressed, use input_size=None, else use input_size=args.input_size
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=10, worker_init_fn=toolbox.worker_init_fn, drop_last=False) for x in ['train', 'val']}

my_metrics = toolbox.Metrics(metric_names='All', num_classes=num_classes)

# from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

out_labels = ["BPR", "FPR", "Healthy", "Not Cocoa", "WBD"]  

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 70)  # Replace with the path to a .ttf file on your system

N = len(dataloaders_dict['val']) + len(dataloaders_dict['train'])
start = time.time()
for phase in ['val']:
	for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
		inputs = inputs.to(device)
		labels = labels.to(device)
		
		# Modify the save path to include a unique filename and file extension
		file_extension = ".jpg"  # or ".jpg" depending on your preference
		save_path = f"/users/jrs596/NotCococa_Fails_ResNet/image_{idx}{file_extension}"
		

		if labels[0] == 3:
			outputs = model(inputs)
			# print(labels[0].item(), outputs[0].argmax().item())
			if labels[0].item() != outputs[0].argmax().item():
				image_to_save = inputs[0].cpu().detach()
				
				pil_img = Image.fromarray((image_to_save.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
				draw = ImageDraw.Draw(pil_img)
				draw.text((0, 0), out_labels[outputs[0].argmax().item()], fill="white", font=font)
				pil_img.save(save_path)
				
				
