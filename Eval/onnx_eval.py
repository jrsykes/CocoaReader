import onnxruntime
import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
#from sklearn import metrics
#import pandas as pd
import time
import numpy as np

model_name = 'CocoaNet18_DN.onnx'

root = '/local/scratch/jrs596/dat/models'
data_dir = "/local/scratch/jrs596/dat/split_cocoa_images"

model = onnxruntime.InferenceSession(os.path.join(root, model_name))

input_size = 750
batch_size = 1
criterion = nn.CrossEntropyLoss()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
	for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
	shuffle=False, num_workers=1) for x in ['train', 'val']}#


def eval_onnx(model, img_loader, input_size):
	lables_list = []
	preds_list = []	

	running_loss = 0.0
	running_corrects = 0
	running_precision = 0
	running_recall = 0
	running_f1 = 0	

	for i in ['val']:#, 'train']:
		for inputs, labels in img_loader[i]:
		#for inputs, labels in dataloaders_dict[i]:
			
			image_data = np.asarray(inputs[0]).astype(np.float32)        
			image_data = np.expand_dims(image_data, 0)

			outputs = model.run([], {'input':image_data})[0]
			#outputs = outputs.flatten()
			outputs = torch.tensor(outputs)

			loss = criterion(outputs, labels)

			_, preds = torch.max(outputs, 1)

		#Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
		#running totals for these metrics by the total number of training or test samples. This controls for 
		#the effect of batch size and the fact that the size of the last batch will not be equal to batch_size
		
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)	
#			stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
#			stats_out = stats['macro avg']
#			running_precision += stats_out['precision']* inputs.size(0)
#			running_recall += stats_out['recall']* inputs.size(0)
#			running_f1 += stats_out['f1-score']* inputs.size(0)#

			for j in labels.data.tolist():
				lables_list.append(j)
			for j in preds.tolist():
				preds_list.append(j)#

		n = len(dataloaders_dict[i].dataset)
		epoch_loss = float(running_loss / n)
		epoch_acc = float(running_corrects.double() / n)
		precision = (running_precision) / n         
		recall = (running_recall) / n        
		f1 = (running_f1) / n	#	
		
	#print('Accuracy: ' + str(round(epoch_acc,3)))
	#print('Precision: ' + str(round(precision,3)))
	#print('Recall: ' + str(round(recall,3)))
	#print('F1: ' + str(round(f1,3)))
		
		print(i)
		print('\n' + '-'*10 + '\nPer class results:')#
		#print(metrics.classification_report(lables_list, preds_list, digits=4))#
		print('loss: ' + str(round(epoch_loss,3)))
		
		
		if i == 'val':
			print('\n' + '-'*10 + '\nConfusion matrix:')
			#print(metrics.confusion_matrix(lables_list, preds_list))#
			#df = pd.DataFrame(metrics.confusion_matrix(lables_list, preds_list))
			#df.to_csv(model_path + '/confusion_matrix.csv')	

	classes = os.listdir(os.path.join(data_dir, 'val'))
	classes.sort()
	number = 0	

	for i in classes:
		print(i, ': ', str(number))
		number += 1	

	print(classes)


start = time.time()

eval_onnx(model, dataloaders_dict, input_size)


print('Time taken: ' + str(time.time()-start))