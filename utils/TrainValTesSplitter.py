import shutil
import os
import random

dat = '/local/scratch/jrs596/dat/FAIGB_combined_hf/'
dest = '/local/scratch/jrs596/dat/FAIGB_combined_hf_split'

healthy_path = '/local/scratch/jrs596/dat/FAIGB_combined/healthy/'
disease_path = '/local/scratch/jrs596/dat/FAIGB_combined/diseased/'

def CopyPlant(destination, img_path):
	n = 0
	for i in os.listdir(img_path):
		for j in os.listdir(os.path.join(img_path,i)):
			n += 1
			source = os.path.join(plant,i,j)
			os.makedirs(os.path.join(destination, 'Plant'), exist_ok = True)
			dest = os.path.join(destination, 'Plant', j + '.jpeg')
			shutil.copy(source, dest)
	return n

#n_not_plant = CopyPlant(destination=dest, img_path = plant)

def CopyNotPlant(destination, img_path, n_not_plant):
	images = os.listdir(img_path)
	random.shuffle(images)
	images = random.sample(images, n_not_plant)	

	for i in images:
		source = os.path.join(img_path, i)
		os.makedirs(os.path.join(destination, 'NotPlant'), exist_ok = True)
		dest = os.path.join(destination, 'NotPlant', i)
		shutil.copy(source, dest)

#CopyNotPlant(destination=dest, img_path=not_plant, n_not_plant=n_not_plant)



def Randomise_Split(dat, destination):
	for class_ in os.listdir(dat):
		images = os.listdir(os.path.join(dat, class_))
		random.shuffle(images)

		if class_ == 'healthy':
			images = random.sample(images, 10000)	

		dat_dict = {'train': images[:int(len(images)*0.8)], 
			'test': images[int(len(images)*0.8):int(len(images)*0.9)], 
			'val': images[int(len(images)*0.9):]}										
		
		for split, im_list in dat_dict.items():
			os.makedirs(os.path.join(destination, split, class_), exist_ok = True)
			for image in im_list:
				source = os.path.join(dat, class_, image)
				dest = os.path.join(destination, split, class_, image)
				shutil.copy(source, dest)

Randomise_Split(dat=dat, destination=dest)

def combine(original_data, disease_path, healthy_path):
	for i in os.listdir(original_data):
		source = os.path.join(original_data, i)

		if 'Diseased' in i:
			print(i)
			for file in os.listdir(source):
				shutil.copy(os.path.join(source,file), os.path.join(disease_path, file))
		else:
			print(i)
			for file in os.listdir(source):
				shutil.copy(os.path.join(source,file), os.path.join(healthy_path, file))
		

#combine(dat, disease_path, healthy_path)