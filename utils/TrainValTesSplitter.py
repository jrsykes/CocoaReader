import shutil
import os
import random

not_plant = '/scratch/staff/jrs596/dat/CLS-LOC_filtered/second_pass/val'
plant = '/scratch/staff/jrs596/dat/ResNetFung50+_images'
dest = '/scratch/staff/jrs596/dat/PlantNotPlant3.2/train_full'


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

CopyNotPlant(destination=dest, img_path=not_plant, n_not_plant=n_not_plant)



def Randomise_Split(destination):
	for class_ in ['Plant', 'NotPlant']:
		images = os.listdir(os.path.join(destination, class_))
		random.shuffle(images)	

		dat_dict = {'train': images[:int(len(images)*0.8)], 
			'test': images[int(len(images)*0.8):int(len(images)*0.9)], 
			'val': images[int(len(images)*0.9):]}										
		
		for split, im_list in dat_dict.items():
			os.makedirs(os.path.join(destination, split, class_), exist_ok = True)
			for image in im_list:
				source = os.path.join(destination, class_, image)
				dest = os.path.join(destination, split, class_, image)
				shutil.copy(source, dest)

#Randomise_Split((destination=dest)