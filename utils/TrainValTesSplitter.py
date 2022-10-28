import shutil
import os
import random
from PIL import Image

src = '/local/scratch/jrs596/dat/PlantPathologyKaggle/dat/unsplit'
dest = '/local/scratch/jrs596/dat/PlantPathologyKaggle/dat'



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

#		images = random.sample(images, 200)	

		dat_dict = {'train': images[:int(len(images)*0.9)], 
			#'test': images[int(len(images)*0.8):int(len(images)*0.9)], 
			'val': images[int(len(images)*0.9):]}										
		
		for split, im_list in dat_dict.items():
			os.makedirs(os.path.join(destination, split, class_), exist_ok = True)
			for image in im_list:
				source = os.path.join(dat, class_, image)
				dest = os.path.join(destination, split, class_, image)
				shutil.copy(source, dest)

Randomise_Split(src, dest)

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

def CopySubset(source, destination):
	images = os.listdir(source)
	images = random.sample(images, 545)	
	for i in images:
		src = os.path.join(source, i)
		dest = os.path.join(destination, i)
		shutil.copy(src, dest)
	

def Image_checker(dir_):
	for i in os.listdir(dir_):
		try:
			image = Image.open(os.path.join(dir_, i))
			print(os.path.join(dir_, i))
		except:
			print('Bad image, deleting')
			#os.remove(os.path.join(dir_, i))


def Size_checker(dir_):
	total = 0
	yes = 0
	for i in os.listdir(dir_):
		total += 1
		image = Image.open(os.path.join(dir_, i))
		width, height = image.size
		if width >= 1120 or height >= 1120:
			#print(os.path.join(dir_, i))
			#print(width, height)
			
			yes += 1
	print(str(yes/total*100), '%')


#root = '/local/scratch/jrs596/dat/FAIGB_combined_hf_split/train'
#print('Diseased: ')
#Size_checker(os.path.join(root, 'diseased'))
#print('Healthy: ')
#Size_checker(os.path.join(root, 'healthy'))

#root = '/local/scratch/jrs596/dat'#

#source = os.path.join(root, 'PlantNotPlant3.3/train_full/Plant')
#destination = os.path.join(root, 'compiled_cocoa_images/NotCocoa')#

##CopySubset(destination, source)

#Image_checker("/local/scratch/jrs596/dat/all_cocoa_images")

#CopySubset(src, dest)