import pandas as pd
import os
import shutil

root = '/local/scratch/jrs596/dat/PlantPathologyKaggle'


train_labels = pd.read_csv(os.path.join(root, 'train.csv'), header=0)
test_labels = pd.read_csv(os.path.join(root, 'test.csv'), header=0)


for index, row in train_labels.iterrows():
	src = os.path.join(root, 'images', row['image_id'] + '.jpg')
	healthy = row['healthy']
	multiple_diseases = row['multiple_diseases']
	rust = row['rust']
	scab = row['scab']

	if healthy == 1:
		dst = os.path.join(root, 'dat/train/healthy')
	elif multiple_diseases == 1:
		dst = os.path.join(root, 'dat/train/multiple_diseases')
	elif rust == 1:
		dst = os.path.join(root, 'dat/train/rust')
	elif scab == 1:
		dst = os.path.join(root, 'dat/train/scab')

	shutil.copy(src, dst)

for index, row in test_labels.iterrows():
	src = os.path.join(root, 'images', row['image_id'] + '.jpg')
	dst = '/local/scratch/jrs596/dat/PlantPathologyKaggle/dat/test'
	shutil.copy(src, dst)