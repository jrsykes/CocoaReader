import torch
import os
from torch import nn
from sklearn import metrics
import pandas as pd
import time
import numpy as np
import shutil


def eval(model, image_datasets, weights_dict):

	print()
	print('Evaluating model...')
	device = torch.device("cuda:0")
	model.eval()

	model = model.to(device)
	criterion = nn.CrossEntropyLoss()

	dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False, num_workers=os.cpu_count()-2)

	lables_list = []
	preds_list = []	

	running_loss = 0.0
	running_auc = 0
	n_classes = len(image_datasets.classes)

	for phase in ['train']:
		running_auc = 0
		running_loss = 0.0
		lables_list = []
		preds_list = []
		for i, (inputs, labels) in enumerate(dataloader,0 ):

			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			_, preds = torch.max(outputs, 1)

			running_loss += loss.item() * inputs.size(0)

			#print file name
			file_name = dataloader.dataset.samples[i][0].split('/')[-1]
			if preds.item() != labels.item():
				weights_dict[file_name] = weights_dict[file_name] * 1.1
			else:
				weights_dict[file_name] = 1
				
			if weights_dict[file_name] > 1.3:
				file_path = dataloader.dataset.samples[i][0]
				os.makedirs(os.path.join("/local/scratch/jrs596/dat/incorrect", str(labels.item())), exist_ok=True)
				shutil.move(file_path, os.path.join("/local/scratch/jrs596/dat/incorrect", str(labels.item()), file_name))
			y = np.zeros(n_classes)
			y[labels] = 1
			x = torch.sigmoid(outputs[0]).cpu().detach().numpy()
			
			fpr, tpr, thresholds = metrics.roc_curve(y, x, pos_label=1)
			auc = metrics.auc(fpr, tpr)
			running_auc += auc
	
			for j in labels.data.tolist():
				lables_list.append(j)
			for j in preds.tolist():
				preds_list.append(j)#

		n = len(dataloader.dataset)
		epoch_loss = float(running_loss / n)
		auc = running_auc / n
	
		#get accuracy
		acc = metrics.accuracy_score(lables_list, preds_list)
		#get f1 score
		F1 = metrics.f1_score(lables_list, preds_list, average='weighted')
		
	return weights_dict, F1, auc, acc, epoch_loss


	
