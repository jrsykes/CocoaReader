import shutil
import os
import random

dir_ = '/local/scratch/jrs596/dat/tiny-imagenet-200/train/'

#classes = os.listdir(dir_)

name = 0
#for i in classes:


dest_path = '/local/scratch/jrs596/dat/NotPlant_TinyIM'

for k in os.listdir('/local/scratch/jrs596/dat/tiny-imagenet-200/train'):
	try:
		os.remove(os.path.join(dir_, k, k + '_boxes.txt'))
	except:
		pass

for k in os.listdir('/local/scratch/jrs596/dat/tiny-imagenet-200/train'):
	images = os.listdir(os.path.join(dir_,k, 'images'))

	for j in images:
		source = os.path.join(dir_, k, 'images', j)
		dest = os.path.join(dest_path, str(name) + '.jpg')
		shutil.copy(source, dest)
		name += 1

