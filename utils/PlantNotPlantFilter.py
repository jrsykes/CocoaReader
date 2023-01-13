import os
import torch
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
import copy
import pickle
from PIL import Image
#from difPy import dif
import matplotlib.pyplot as plt
import time


model_name = "PlantNotPlant_SemiSup"
root = '/local/scratch/jrs596/dat'
image_out_dir = os.path.join(root, 'ElodeaProject/FasterRCNN_output/PNP_filtered')


data_dir = os.path.join(root, "ElodeaProject/FasterRCNN_output/Images_out")
#data_dir = '/local/scratch/jrs596/dat/PlantNotPlant3.3/train_full'
#data_dir = os.path.join(root, "test2/images")
#model_path = os.path.join(root, 'models')
model_path = '/local/scratch/jrs596/dat/models'
device = torch.device("cuda")

def prep_model():
	  
	# Number of classes in the dataset
	num_classes = 2


	model = models.resnet18(weights=None)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, num_classes)	

	pretrained_model_path = os.path.join(model_path, model_name + '.pkl')
	pretrained_model_wts = pickle.load(open(pretrained_model_path, "rb"))	

	weights = copy.deepcopy(pretrained_model_wts['model'])	

	###############
	#Remove 'module.' from layer names
	new_keys = []
	for key, value in weights.items():
	    new_keys.append(key.replace('module.', ''))
	for i in new_keys:
	    weights[i] = weights.pop('module.' + i)    	

	##############
	model.load_state_dict(weights)
	model.eval()
	model.to(device)
	return model


##############################################
# Filter out non .JPEG and corrupt image files


def filter_corrupt_files():
	dest = os.path.join(image_out_dir, 'corrupt') 
	os.makedirs(dest, exist_ok = True)
	index = len(os.listdir(dest))
	for class_ in os.listdir(data_dir):
		for image in os.listdir(os.path.join(data_dir, class_)):
			file_path = os.path.join(data_dir,class_,image)
			try:
				im = Image.open(file_path, formats=['JPEG'])
			except:
				shutil.move(file_path, os.path.join(dest, str(index) + '.jpg'))
				index += 1
				print(file_path)

#filter_corrupt_files()

###############################################
## Delete duplicate images for each class
def delete_duplicates():
	for i in os.listdir(data_dir):
		search = dif(os.path.join(data_dir, i), delete=True, silent_del=True)

#delete_duplicates()

##############################################
# Filter out non-plant images with Plant-NotPlant CNN

def plant_notplant_filter():
	model = prep_model()
	input_size = 224
	transform = transforms.Compose([
	    transforms.Resize((input_size,input_size)),
	    transforms.ToTensor(),
	    ])	

	dataset = datasets.ImageFolder(data_dir, transform=transform)
	loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)
	classes = sorted(os.listdir(data_dir))
	
	for i, (inputs, labels) in enumerate(loader):
		class_ = classes[labels.item()]
		source, _ = loader.dataset.samples[i]
		inputs = inputs.to(device)
		outputs = model(inputs)
		outputs = torch.sigmoid(outputs)			
		print(i)
		print(outputs)
		print()
		# [0][1] = Plant
		# [0][0] = NotPlant
		
		# out_path = "/local/scratch/jrs596/dat/ElodeaProject/FasterRCNN_output/PNP_filtered/"

		# #If model predicts "plant" 
		# if outputs[0][0].item() > outputs[0][1].item():# outputs[0][0].item():
			
		# 	dest = os.path.join(out_path, 'Entangled')
		# 	os.makedirs(dest, exist_ok=True)
		# 	shutil.copy(source, dest + '/' + str(i) + '.jpeg')

		# #If model predicts "not plant"
		# else: #outputs[0][1].item() > 0.9: #outputs[0][1].item() * 1.01:
		# 	dest = os.path.join(out_path, 'NotEntangled')
		# 	os.makedirs(dest, exist_ok=True)

		# 	shutil.copy(source, dest + '/' + str(i) + '.jpeg')



#%%
		#If model is unsure
#		else:
#			print('\n', source, '\n')
#			plt.imshow(inputs[0].cpu().permute(1, 2, 0))
#			plt.draw()
#			plt.pause(2)
#			plt.close()
#			answer = input("\nIs this a plant, y or n? Or press a to see again. ")#
#		
#			if answer == 'a':
#				print('\n', source, '\n')
#				plt.imshow(inputs[0].cpu().permute(1, 2, 0))
#				plt.draw()
#				plt.pause(2)
#				plt.close()
#				answer = input("\nIs this a plant, y or n? Or press a to see again. ")##

#			elif answer == 'n':
#				print('Deleting image')
#				dest = os.path.join(image_out_dir, class_ + str(time.time()) + '.jpg')
#				print('Confidence, not plant: ' , str(outputs[0][0].item()))
#			
#			else:
#				dest = os.path.join(root, 'Forestry_ArableImages_GoogleBing_Final', class_, class_ + str(time.time()) + '.jpg')
#				print('Keeping image')
#				print(dest)
#				print('Confidence, plant: ' , str(outputs[0][1].item()))

		#shutil.copy(source, dest)



plant_notplant_filter()


