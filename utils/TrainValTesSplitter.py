import shutil
import os
import random


dir_ = '/local/scratch/jrs596/dat/PlantNotPlant_TinyIM+VAE_Filtered'

classes = os.listdir(dir_)


for i in classes:
	images = os.listdir(os.path.join(dir_, i))
	random.shuffle(images)
	dat_dict = {'train': images[:int(len(images)*0.8)], 
		'test': images[int(len(images)*0.8):int(len(images)*0.9)], 
		'val': images[int(len(images)*0.9):]}


	for key, value in dat_dict.items():
		dest_path = '/local/scratch/jrs596/dat/PlantNotPlant_TinyIM+VAE_Filtered_split'
		os.makedirs(os.path.join(dest_path , key, i), exist_ok = True)
		for j in value:
			source = os.path.join(dir_, i, j)
			dest = os.path.join(dest_path, key, i, j)
			shutil.copy(source, dest)


