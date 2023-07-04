#%%
import shutil
import os
import random
from PIL import Image
import time


#%%
# src = '/local/scratch/jrs596/dat/ElodeaProject/FasterRCNN_output/Rudders'
# dest = '/local/scratch/jrs596/dat/ElodeaProject/FasterRCNN_output/Rudders_split'


def Organise_Ecuador_Images(root):
	count = 0
	other = 0
	dumby_imgs = ["left1.png", "1.JPG", "20230123_161149.jpg", "PXL_20230122_115658050.jpg", "PXL_20230203_115351448.jpg", "PXL_20230122_093124778.jpg"]
	for d in os.listdir(os.path.join(root, "HandFiltered_Ecuador_images")):
		if d != "Demo_files" and d != "ReadMe.txt":
			for f in os.listdir(os.path.join(root, "HandFiltered_Ecuador_images", d)):
				img_dir = os.path.join(root, "HandFiltered_Ecuador_images", d, f)
				image_dirs = os.listdir(img_dir)
				for z in image_dirs:
					dir = os.path.join(root, "HandFiltered_Ecuador_images", d, f, z)
					images = os.listdir(dir)
					dest = os.path.join(root, "EcuadorImages_All")
					# if z.endswith("Temprano"):
					# 	dest = os.path.join(root, "EcuadorImages_EL_FullRes/Early")
					# elif z.endswith("Unsure"):
					# 	dest = os.path.join(root, "EcuadorImages_EL_FullRes/Unsure")
					# elif z.endswith("Tarde"):
					# 	dest = os.path.join(root, "EcuadorImages_EL_FullRes/Late")
					# elif z == "Sana":
					# 	dest = os.path.join(root, "EcuadorImages_EL_FullRes/Late")
					# else:
					# 	other += len(images)
					# 	print("End incorrect")
					# 	print(os.path.join(root, "HandFiltered_Ecuador_images", d, f, z))
					# 	print()

					if z.startswith("Monilia"):
						class_ = "FPR"
					elif z.startswith("Fitoptora"):
						class_ = "BPR"
					elif z.startswith("Escoba"):
						class_ = "WBD"
					elif z.startswith("Sana"):
						class_ = "Healthy"
					elif z.startswith("Virus"):
						class_ = "Virus"
					else:
						print("Start incorrect")
						print(os.path.join(root, "HandFiltered_Ecuador_images", d, f, z))
						print()
					os.makedirs(os.path.join(dest, class_), exist_ok = True)
					for i in images:
						if i not in dumby_imgs:
							count += 1
							#shutil.copy(os.path.join(dir, i), os.path.join(dest, class_, str(time.time()) + i))
							#make symbolic link
							os.symlink(os.path.join(dir, i), os.path.join(dest, class_, str(time.time()) + i))
							#open method used to open different extension image file
							#im = Image.open(os.path.join(dir, i)) 
							#im1 = im.resize((1200,1200))
							#im1.save(os.path.join(dest, class_, str(time.time()) + i))
						elif i in dumby_imgs:
							pass
						else:
							other += 1
							print(i)
	
	print("Image count: " + str(count))    
	print("Filtered image count: " + str(other))   
	print("Total image count: " + str(other + count))   

#root = "/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/Ecuador_data"
#Organise_Ecuador_Images(root)

def CopyPlant(destination, img_path):
	n = 0
	for i in os.listdir(img_path):
		for j in os.listdir(os.path.join(img_path,i)):
			n += 1
			source = os.path.join('plant',i,j)
			os.makedirs(os.path.join(destination, 'Plant'), exist_ok = True)
			dest = os.path.join(destination, 'Plant', j + '.jpeg')
			shutil.copy(source, dest)
	return n


def CopyNotPlant(destination, img_path, n_not_plant):
	images = os.listdir(img_path)
	random.shuffle(images)
	images = random.sample(images, n_not_plant)	

	for i in images:
		source = os.path.join(img_path, i)
		os.makedirs(os.path.join(destination, 'NotPlant'), exist_ok = True)
		dest = os.path.join(destination, 'NotPlant', i)
		shutil.copy(source, dest)




def Randomise_Split(root, destination):
	for class_ in os.listdir(root):
		images = os.listdir(os.path.join(root, class_))
		random.shuffle(images)
		#images = images[:800]

		dat_dict = {'train': images[:int(len(images)*0.9)], 
			#'test': images[int(len(images)*0.8):int(len(images)*0.9)], 
			'val': images[int(len(images)*0.9):]}										
		
		for split, im_list in dat_dict.items():
			#os.makedirs(os.path.join(root, split, class_), exist_ok = True)
			
			for image in im_list:
				source = os.path.join(root, class_, image)
				dest = os.path.join(destination, split, class_)
				os.makedirs(dest, exist_ok = True)
				#open image and compress to 330x330 pixels
				im = Image.open(os.path.join(source))
				im1 = im.resize((494,494))
				im1.save(os.path.join(dest, image))

				#shutil.copy(source, dest)
				#os.symlink(source, os.path.join(dest, image))

root = '/local/scratch/jrs596/dat/FAIGB_PNPFiltered_HandFiltered_PinopsidaCollapsed'
destination = '/local/scratch/jrs596/dat/FAIGB_494'

Randomise_Split(root, destination)

def compress_copy(root, destination):
	for class_ in os.listdir(root):
		images = os.listdir(os.path.join(root, class_))

		for image in images:
			source = os.path.join(root, class_, image)
			dest = os.path.join(destination, class_)
			os.makedirs(os.path.join(dest), exist_ok = True)
			#open image and compress to 330x330 pixels
			im = Image.open(os.path.join(source))
			im1 = im.resize((330,330))
			im1.save(os.path.join(dest, image))
		

#source = '/local/scratch/jrs596/dat/EcuadorWebImages_EasyDif_FinalClean/Unsure'
#destination = '/local/scratch/jrs596/dat/EcuadorWebImages_EasyDif_FinalClean_SplitCompress/Unsure'

#compress_copy(source, destination)

def combine(original_data, disease_path, healthy_path):
	os.makedirs(disease_path, exist_ok = True)
	os.makedirs(healthy_path, exist_ok = True)
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
		


def CopySubset(source, destination):
	images = os.listdir(source)
	images = random.sample(images, 400)	
	for i in images:
		src = os.path.join(source, i)
		dest = os.path.join(destination, i)
		shutil.copy(src, dest)
	
#CopySubset(source = '/local/scratch/jrs596/dat/all_cocoa_images', 
#	   destination = '/local/scratch/jrs596/dat/subset_cocoa_images')

def CopySubsetForNotCacoa(dat, destination):
	#list all images in subdirectories of dat
	images = []
	for dir_ in os.listdir(dat):
		for image in os.listdir(os.path.join(dat, dir_)):
			images.append(os.path.join(dir_, image))

	print("Number of images: ", len(images))
	images = random.sample(images, 1000)	

	train = images[:int(len(images)*0.8)]
	val = images[int(len(images)*0.8):int(len(images)*0.9)]
	test = images[int(len(images)*0.9):]

	for i in train:
		#split file path
		dir_, file = os.path.split(i)
		source = os.path.join(dat, i)
		dest = os.path.join(destination, 'train', 'NotCocoa', file)
		shutil.copy(source, dest)
	for i in val:
		dir_, file = os.path.split(i)
		source = os.path.join(dat, i)
		dest = os.path.join(destination, 'val','NotCocoa', file)
		shutil.copy(source, dest)
	for i in test:
		dir_, file = os.path.split(i)
		source = os.path.join(dat, i)
		dest = os.path.join(destination, 'test','NotCocoa', file)
		shutil.copy(source, dest)
	

def Image_checker(dir_):
	for i in os.listdir(dir_):
		try:
			image = Image.open(os.path.join(dir_, i))
			#print(os.path.join(dir_, i))
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


def Randomise_combine_subset(dat, destination):
	for dir_ in os.listdir(dat):
		images = os.listdir(os.path.join(dat, dir_))
		random.shuffle(images)

		if len(images) > 50:
			images = random.sample(images, 50)	

		for i in images:
			source = os.path.join(dat, dir_, i)
			dest = os.path.join(destination, str(time.time()) + '.jpeg')
			shutil.copy(source, dest)								
		

def cocoa_image_complier(dat, destination):
	for dir_ in os.listdir(dat):
		images = os.listdir(os.path.join(dat, dir_))
		random.shuffle(images)

		train = images[:int(len(images)*0.8)]
		val = images[int(len(images)*0.8):int(len(images)*0.9)]
		test = images[int(len(images)*0.9):]

	
		for i in train:
			source = os.path.join(dat, dir_, i)
			dest_ = os.path.join(destination, 'train', dir_, i)
			shutil.copy(source, dest_)
		for i in val:
			source = os.path.join(dat, dir_, i)
			dest_ = os.path.join(destination, 'val', dir_, i)
			shutil.copy(source, dest_)
		for i in test:
			source = os.path.join(dat, dir_, i)
			dest_ = os.path.join(destination, 'test', dir_, i)
			shutil.copy(source, dest_)


#%%

#cocoa_image_complier(dat="/local/scratch/jrs596/dat/EcuadorImages_EL_LowRes/Combined", destination="/local/scratch/jrs596/dat/split_cocoa_images2")
# %%
