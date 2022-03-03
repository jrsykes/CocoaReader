import os
import numpy as np
import shutil

path = '/local/scratch/jrs596/dat/ResNetFung50+_images_organised/train/'

set_sizes = []
for i in os.listdir(path):
	set_sizes.append(len(os.listdir(path + i)))

count = 0
set_dict = {}
for i in os.listdir(path):
	max_ = round(np.max(set_sizes))
	set_dict[i] = round(max_/set_sizes[count]) 
	count += 1

path2 = '/local/scratch/jrs596/dat/ResNetFung50+_images_organised_balanced/train/'


for value, key in set_dict.items():
	os.mkdir(path2 + value)
	count = 0
	for i in range(key):
		files = os.listdir(path + value)
		for j in files:
			shutil.copyfile(path + value + '/' + j, path2 + value + '/' + str(count) + j)
			count += 1
	
