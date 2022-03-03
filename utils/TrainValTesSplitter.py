import shutil
import os
import random

dir_ = '/local/scratch/jrs596/dat/compiled_cocoa_images/'

classes = ['FPR', 'Healthy', 'BPR', 'WBD']

for i in classes:
	images = os.listdir(dir_ + i)
	random.shuffle(images)
	train_set = images[:int(len(images)*0.9)]
	test_set = images[int(len(images)*0.9):]
	
	os.makedirs(dir_ + 'CrossVal_split/train/' + i, exist_ok = True)
	os.makedirs(dir_ + 'CrossVal_split/test/' + i, exist_ok = True)
	
	for j in train_set:
		source = dir_ + i + '/' + j
		dest = dir_ + 'CrossVal_split/train/' + i + '/' + j
		shutil.copy(source, dest)
		
	for j in test_set:
		source = dir_ + i + '/' + j
		dest = dir_ + 'CrossVal_split/test/' + i + '/' + j
		shutil.copy(source, dest)

