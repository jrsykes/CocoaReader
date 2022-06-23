import shutil
import os
import random

dir_ = '/local/scratch/jrs596/dat/Plant'
dest_path = '/local/scratch/jrs596/dat/PlantNotPlant'

#name = 0
#for k in os.listdir(dir_):
##	try:
##		os.remove(os.path.join(dir_, k, k + '_boxes.txt'))
##	except:
##		pass#

##	for i in ['Plant', 'NotPlant']:
##		images = os.listdir(os.path.join(dir_,k, i))#

#	for j in os.listdir(os.path.join(dir_,k)):
#		source = os.path.join(dir_, k, j)
#		dest = os.path.join(dest_path, j + '.jpg')
#		shutil.copy(source, dest)
		#name += 1


images = os.listdir(dir_)
random.shuffle(images)
dat_dict = {'train': images[:int(len(images)*0.8)], 
	'test': images[int(len(images)*0.8):int(len(images)*0.9)], 
	'val': images[int(len(images)*0.9):]}

for key, value in dat_dict.items():
	
	#os.makedirs(os.path.join(dest_path , key, i), exist_ok = True)
	for j in value:
		dest = os.path.join(dest_path, key, 'Plant', j)
		source = os.path.join(dir_, j)
		shutil.copy(source, dest)