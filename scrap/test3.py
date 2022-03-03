from torchvision import datasets, models, transforms
import os
import torch
import random
import shutil
data_dir = "/local/scratch/jrs596/dat/compiled_cocoa_images/CrossVal_split/train_val/" 

temp_dir = '/local/scratch/jrs596/dat/compiled_cocoa_images/CrossVal_split/epoch_randomisation/'

try:
	shutil.rmtree(temp_dir + 'train')
	shutil.rmtree(temp_dir + 'train')
except:
	pass

for i in ['FPR', 'WBD', 'Healthy', 'BPR']:
	image_list = os.listdir(data_dir + i)
	random.shuffle(image_list)
	for j in ['train', 'val']:
		os.makedirs(temp_dir + j + '/' + i, exist_ok = True)
		if j == 'train':
			sample = image_list[:int(len(image_list)*0.9)]
			for k in sample:
				source = data_dir + i + '/' + k
				dest = temp_dir + j + '/' + i + '/' + k
				shutil.copy(source, dest)
		else:
			sample = image_list[int(len(image_list)*0.9):]
			for k in sample:
				source = data_dir + i + '/' + k
				dest = temp_dir + j + '/' + i + '/' + k
				shutil.copy(source, dest)






#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#
#

#def random_data_gen():#

#	data_transforms = {
#	    'train': transforms.Compose([
#	        transforms.RandomResizedCrop(input_size),
#	        transforms.RandomHorizontalFlip(),
#	        transforms.ToTensor(),
#	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#	    ]),
#	    'val': transforms.Compose([
#	        transforms.Resize(input_size),
#	        transforms.CenterCrop(input_size),
#	        transforms.ToTensor(),
#	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#	    ]),
#	}	#

#	data = datasets.ImageFolder(data_dir)#

#	n = len(data.samples)	#

#	image_datasets = torch.utils.data.random_split(data, [round(n*0.1), round(n*0.9)], 
#		generator=torch.Generator())#

#	image_datasets = datasets.ImageFolder(image_datasets)
#	
##	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
##		data_transforms[x]) for x in ['train_val']}	#

#	
#	##

#	#dataloaders_dict = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, 
#	#	shuffle=True, num_workers=4)	#

#	return image_datasets#

#dataloaders = random_data_gen()#

##for inputs, labels in dataloaders['train']:
# #   print(inputs)# = inputs.to(device)
#    #labels = labels.to(device)#
#

#print(dataloaders)