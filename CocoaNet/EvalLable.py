import torch
import os
from torch import nn
from sklearn import metrics
import pandas as pd
import time
import numpy as np
import shutil
from torchvision import transforms, datasets



def eval(model, moved_count, input_size):
	print()
	print('Evaluating model...')
	device = torch.device("cuda:0")
	model.eval()
	model = model.to(device)
	
	data_transforms = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()])
    
	image_dataset = datasets.ImageFolder('/local/scratch/jrs596/dat/EcuadorImages_EL_LowRes_split/Early', data_transforms)
	dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count()-2, drop_last=True)

	for i, (inputs, labels) in enumerate(dataloader,0 ):
		inputs = inputs.to(device)
		labels = labels.to(device)
		
		#split batch into list of tensors
		inputs = torch.split(inputs, 1, dim=0)
		token_size = int(input_size / 4)
		x, y, h ,w = 0, 0, token_size, token_size
		ims = []
		for t in inputs:
            #crop im to 16 non-overlapping 277x277 tensors
			for i in range(4):
				for j in range(4):
					t.squeeze_(0)
					im1 = t[:, x:x+h, y:y+w]
					ims.append(im1)
					y += w
				y = 0
				x += h
			x, y, h ,w = 0, 0, token_size, token_size
			#convert list of tensors into a batch tensor of shape (512, 3, 277, 277
		inputs = torch.stack(ims, dim=0)

		outputs = model(inputs).mean(dim=0).unsqueeze(0)
		_, preds = torch.max(outputs, 1)
		
		if labels.item() == 0:
			class_ = 'BPR'
		elif labels.item() == 1:
			class_ = 'FPR'
		elif labels.item() == 2:
			class_ = 'Healthy'
		elif labels.item() == 3:
			class_ = 'WBD'
		
		file_name = dataloader.dataset.samples[i][0].split('/')[-1]
		#get model confidence for prediction
		confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds.item()].item()
		if confidence >= 0.98:
			file_path = dataloader.dataset.samples[i][0]
			dest = os.path.join("/local/scratch/jrs596/dat/EcuadorImage_LowRes_17_03_23_SureUnsure/Sure/train", class_, file_name)

			shutil.move(file_path, dest)
			
			moved_count += 1
	print('\n Number of early images labled: ', str(moved_count), '\n')
	return moved_count