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

data_dir = "/local/scratch/jrs596/dat/IR_RGB_Comp_data/IR_split_400"

model_path = '/local/scratch/jrs596/dat/models/DisNet-pico-IR_ArchSweepBest.pt'
quantized = False

n_classes = len(os.listdir(os.path.join(data_dir, 'train')))

input_size = 400
batch_size = 1
criterion = nn.CrossEntropyLoss()

data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        #transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ])
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
	for x in ['train', 'val']}#, 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
	shuffle=False, num_workers=6) for x in ['train', 'val']}#, 'test']}#






def eval(model, dataloaders_dict):
	lables_list = []
	preds_list = []	

	running_loss = 0.0
	running_corrects = 0
	running_precision = 0
	running_recall = 0
	running_f1 = 0	
	running_auc = 0

	#df = pd.DataFrame(columns=['image_id','healthy','multiple_diseases','rust','scab'])
	for phase in ['train', 'val']:
		running_auc = 0
		running_loss = 0.0
		lables_list = []
		preds_list = []
		for i, (inputs, labels) in enumerate(dataloaders_dict[phase],0 ):
			#filename, _ = dataloaders_dict[phase].dataset.samples[i]
			#head_tail = os.path.split(filename)

			# filename = head_tail[1][:-4]
			# input_img_ = inputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
			# input_img_ = input_img_.astype(np.uint8)
			# tmp_out = "/local/scratch/jrs596/dat/ElodeaProject/croped_tensors"
			# cv2.imwrite(os.path.join(tmp_out, filename), input_img_)

			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			_, preds = torch.max(outputs, 1)

		#Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
		#running totals for these metrics by the total number of training or test samples. This controls for 
		#the effect of batch size and the fact that the size of the last batch will not be equal to batch_size
		
			running_loss += loss.item() * inputs.size(0)

			#Convert labels to 1 hot matrix
			y = np.zeros(n_classes)
			y[labels] = 1
			
			x = torch.sigmoid(outputs[0]).cpu().detach().numpy()
			
			#out = [filename] + list(x)
			#df.loc[len(df)] = out

#			for i in x:
#				out = out + ',' + str(round(i,2)) + '\n'


			fpr, tpr, thresholds = metrics.roc_curve(y, x, pos_label=1)
			auc = metrics.auc(fpr, tpr)
			running_auc += auc
	
			for j in labels.data.tolist():
				lables_list.append(j)
			for j in preds.tolist():
				preds_list.append(j)#

		n = len(dataloaders_dict[phase].dataset)
		epoch_loss = float(running_loss / n)
		auc = running_auc / n
	
		print(phase)
		print('\n' + '-'*10 + '\nPer class results:')#
		print(metrics.classification_report(lables_list, preds_list, digits=4, zero_division=True))#
		print('loss: ' + str(round(epoch_loss,4)))
		print('AUC: ' + str(round(auc, 4)))
		
		print('\n' + '-'*10 + '\nConfusion matrix:')
		print(metrics.confusion_matrix(lables_list, preds_list))
		print()


	# classes = os.listdir(os.path.join(data_dir, 'val'))
	# classes.sort()
	# number = 0	

	# for i in classes:
	# 	print(i, ': ', str(number))
	# 	number += 1	

	# print()
	# print(classes)
	# print()

	# print(df)
	# df.to_csv('/local/scratch/jrs596/dat/PlantPathologyKaggle/out.csv', index=False)


def quant_eval(model, img_loader):
    elapsed = 0
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    #return elapsed


def quant_eval2(model, img_loader):
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

			outputs = model(inputs)
			loss = criterion(outputs, labels)
			_, preds = torch.max(outputs, 1)

		#Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
		#running totals for these metrics by the total number of training or test samples. This controls for 
		#the effect of batch size and the fact that the size of the last batch will not be equal to batch_size
		
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)	
			stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
			stats_out = stats['macro avg']
			running_precision += stats_out['precision']* inputs.size(0)
			running_recall += stats_out['recall']* inputs.size(0)
			running_f1 += stats_out['f1-score']* inputs.size(0)#

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
		print(metrics.classification_report(lables_list, preds_list, digits=4))#
		print('loss: ' + str(round(epoch_loss,3)))
		
		
		if i == 'val':
			print('\n' + '-'*10 + '\nConfusion matrix:')
			print(metrics.confusion_matrix(lables_list, preds_list))#
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



if quantized == False:
	model = torch.load(model_path)
else:
	model = torch.jit.load(model_path)

model.eval()

print(model)

if quantized == False:
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")

model = model.to(device)

if quantized == True:
	#quant_eval(model, dataloaders_dict['val'])
	quant_eval2(model, dataloaders_dict)
else:
	eval(model, dataloaders_dict)

print('Time taken: ' + str(time.time()-start))



