#%%
import shutil
import os
import random
from PIL import Image
import time
from sklearn.model_selection import StratifiedKFold
import glob
import numpy as np

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
		
		if class_ != 'ReadMe.md':
			print("\nProcessing class: ", class_)
			images = os.listdir(os.path.join(root, class_))
			random.shuffle(images)
			print("Number of images: ", len(images))
			if len(images) > 100: 
				print("90:10 split")
				dat_dict = {'train': images[:int(len(images)*0.9)], 
					'val': images[int(len(images)*0.9):]}
			elif len(images) > 10:
				print("80:20 split")
				dat_dict = {'train': images[:int(len(images)*0.8)],
				   'val': images[int(len(images)*0.8):]}
			else:
				print("50:50 split")
				dat_dict = {'train': images[:int(len(images)*0.5)],
				   'val': images[int(len(images)*0.5):]}

			for split, im_list in dat_dict.items():
				print("Processing split: ", split)

				for image in im_list:
					source = os.path.join(root, class_, image)
					dest = os.path.join(destination, split, class_)
					os.makedirs(dest, exist_ok = True)
					
					try:
						im = Image.open(source)
						# Apply EXIF orientation
						im = apply_exif_orientation(im)
						im1 = im.resize((600,600))
						im1.save(os.path.join(dest, image))
					except Exception as e:
						print("Bad image or error processing image:", e)
						print(source)
						# Consider logging error and continuing instead of exiting
						# exit()

def apply_exif_orientation(image):
	try:
		exif = image._getexif()
		orientation_key = 274  # cf. EXIF 2.2 specification
		if exif and orientation_key in exif:
			orientation = exif[orientation_key]
			rotations = {
				3: Image.ROTATE_180,
				6: Image.ROTATE_270,
				8: Image.ROTATE_90
			}
			if orientation in rotations:
				return image.transpose(rotations[orientation])
	except AttributeError:
		pass  # No EXIF data or couldn't find orientation data
	return image


root = '/users/jrs596/Jun_Cocoa_Project/Cocoa_leaf_data'
destination = '/users/jrs596/Jun_Cocoa_Project/Cocoa_leaf_data_split'

# Randomise_Split(root, destination)

# def NonEasyDif_Spliter(root, destination):
# 	for class_ in os.listdir(root):
		
# 		if class_ != 'ReadMe.md':
# 			print("\nProcessing class: ", class_)
# 			images = os.listdir(os.path.join(root, class_))
# 			random.shuffle(images)

# 			val_images = os.listdir(os.path.join(destination, 'val', class_))

# 			for img in images:
# 				if img not in val_images:

# 					source = os.path.join(root, class_, img)
# 					dest = os.path.join(destination, 'train', class_)
# 					os.makedirs(dest, exist_ok = True)
# 					shutil.copy(source, dest)


# root = '/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_FinalClean_Compress500'
# destination = '/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_FinalClean_Compress500_split'

# NonEasyDif_Spliter(root, destination)


def FAIGB_Spliter(root, destination):
	TrainValimages = []
	for split in os.listdir(root):
		for class_ in os.listdir(os.path.join(root, split)):
			images = os.listdir(os.path.join(root, split, class_))
			full_paths = [os.path.join(root, split, class_, image) for image in images]
			TrainValimages.extend(full_paths)  

	random.shuffle(TrainValimages)
	   
	images_dict = {'train': TrainValimages[:1800], 'val': TrainValimages[len(TrainValimages)-200:]}
 
	for split, im_list in images_dict.items():
		dest = os.path.join(destination, split)
		os.makedirs(dest, exist_ok = True)
		for image in im_list:
			shutil.copy(image, os.path.join(dest, os.path.basename(image)))

root = '/users/jrs596/scratch/dat/FAIGB/FAIGB_700_30-10-23_split'
destination = '/users/jrs596/scratch/dat/Ecuador/NotCocoa'

# FAIGB_Spliter(root, destination)

def compress_copy(root, destination):
	os.makedirs(destination, exist_ok = True)
	for split in os.listdir(root):
		print("Processing split: ", split)
		for class_ in os.listdir(os.path.join(root, split)):
			print("Processing class: ", class_)
			images = os.listdir(os.path.join(root, split, class_))

			for image in images:
				source = os.path.join(root, split, class_, image)
				dest = os.path.join(destination, split, class_)
				os.makedirs(os.path.join(dest), exist_ok = True)
				#open image and compress to 330x330 pixels
				im = Image.open(os.path.join(source))
				im1 = im.resize((700,700))
				im1.save(os.path.join(dest, image))
		

source = '/users/jrs596/scratch/dat/FAIGB/FAIGB_FullRes_30-10-23_split'
destination = '/users/jrs596/scratch/dat/FAIGB/FAIGB_700_30-10-23_split'

compress_copy(source, destination)

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



def resize_and_save_image(image_path, output_path):
	try:
		with Image.open(image_path) as img:
			img = img.resize((700, 700))
			img.save(output_path)
			print(f"Image {image_path} resized and saved to {output_path}")
	except Exception as e:
		print(f"Error processing {image_path}: {e}")

def process_directory(input_directory, output_directory):
	for root, dirs, files in os.walk(input_directory):
		for file in files:
			if file.lower().endswith(('.png', '.jpg', '.jpeg')):
				input_image_path = os.path.join(root, file)
				relative_subdir = os.path.relpath(root, input_directory)
				output_subdir = os.path.join(output_directory, relative_subdir)
				os.makedirs(output_subdir, exist_ok=True)
				output_image_path = os.path.join(output_subdir, file)
				resize_and_save_image(input_image_path, output_image_path)


# process_directory(input_directory, output_directory)

def cross_validation_split(dataset_dir, output_dir):
	# Creating the output directory if it doesn't exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# Collect all image paths and their corresponding class labels
	image_paths = []
	labels = []
	
	# List of possible JPEG extensions
	jpeg_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
	
	for class_dir in os.listdir(dataset_dir):
		if os.path.isdir(os.path.join(dataset_dir, class_dir)):
			# Initialize an empty list for the class to hold paths from all extensions
			class_paths = []
			# Search for each extension and combine the results
			for extension in jpeg_extensions:
				pattern = os.path.join(dataset_dir, class_dir, extension)
				class_paths.extend(glob.glob(pattern))
	
			# Extend the image_paths and labels lists with the results
			image_paths.extend(class_paths)
			labels.extend([class_dir] * len(class_paths))

	# Convert labels to a numerical format for stratified splitting
	unique_labels = np.unique(labels)
	label_to_index = {label: index for index, label in enumerate(unique_labels)}
	numerical_labels = [label_to_index[label] for label in labels]

	# Stratified K-Fold instantiation for ten-fold cross-validation
	skf = StratifiedKFold(n_splits=10)

	# Splitting the dataset
	for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, numerical_labels)):
		print(f'Preparing fold {fold}')
		fold_dir = os.path.join(output_dir, f'fold_{fold}')
		train_dir = os.path.join(fold_dir, 'train')
		val_dir = os.path.join(fold_dir, 'val')

		# Creating directories for the current fold
		for path in [train_dir, val_dir]:
			if not os.path.exists(path):
				os.makedirs(path)

		# Function to copy and resize files to the respective directories
		def copy_and_resize_images(indices, target_dir):
			for i in indices:
				image_path = image_paths[i]
				class_label = labels[i]
				target_class_dir = os.path.join(target_dir, class_label)
				if not os.path.exists(target_class_dir):
					os.makedirs(target_class_dir)
				# Open and resize the image
				with Image.open(image_path) as img:
					img_resized = img.resize((600,600))
					# Construct the target path and save the resized image
					target_path = os.path.join(target_class_dir, os.path.basename(image_path))
					img_resized.save(target_path)

		# Copy training and validation images to their respective directories
		copy_and_resize_images(train_idx, train_dir)
		copy_and_resize_images(val_idx, val_dir)

	print("Dataset splitting complete.")

# # The root directory where the dataset is located
# dataset_dir = '/users/jrs596/scratch/dat/IR_RGB_Comp_data/compiled_IR/'

# # The directory where the split dataset will be saved
# output_dir = '/users/jrs596/scratch/dat/IR_RGB_Comp_data/IR_CrossVal_600'

# cross_validation_split(dataset_dir, output_dir)