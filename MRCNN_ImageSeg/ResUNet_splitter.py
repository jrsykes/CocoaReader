import os

base_path = '/local/scratch/jrs596/ResUNet/dat/'

images = sorted(os.listdir(base_path + 'JPEGImages'))

masks = sorted(os.listdir(base_path + 'SegmentationClass'))



dat = {}
for i in images:
	file_name = (os.path.splitext(i)[0])
	if file_name + '.png' in masks:
		dat[i] = file_name + '.png'
		

train = {k: dat[k] for k in list(dat)[:round(len(dat)*0.8)]}
test = {k: dat[k] for k in list(dat)[round(len(dat)*0.8):round(len(dat)*0.9)]}
val = {k: dat[k] for k in list(dat)[round(len(dat)*0.9):round(len(dat))]}


with open(base_path + 'train.txt', 'w') as f:
	for key, value in train.items():
		f.write(base_path + 'JPEGImages/' + key + ', ' + base_path + 'SegmentationClass/' + value + '\n')

with open(base_path + 'test.txt', 'w') as f:
	for key, value in test.items():
		f.write(base_path + 'JPEGImages/' + key + ', ' + base_path + 'SegmentationClass/' + value + '\n')

with open(base_path + 'val.txt', 'w') as f:
	for key, value in val.items():
		f.write(base_path + 'JPEGImages/' + key + ', ' + base_path + 'SegmentationClass/' + value + '\n')