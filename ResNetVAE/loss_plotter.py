import pandas as pd
from matplotlib import pyplot as plt
import os
import statistics
from math import sqrt
from scipy.stats import norm
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import shutil

print('Fix bug. Scrap images in wrong class folder')
exit(0)
img_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean/'

df = pd.read_csv('/local/scratch/jrs596/ResNetVAE/results/losses.csv', header=0)

classes = tuple(os.listdir(img_dir))

dataset = torchvision.datasets.ImageFolder(img_dir)



shutil.rmtree('/local/scratch/jrs596/dat/scrap_imgs')
key_list = list(dataset.class_to_idx.keys())
val_list = list(dataset.class_to_idx.values())

arr = df['loss'].to_numpy()
ci = norm(*norm.fit(arr)).interval(0.90)

for j in classes:
	print(j)
	
	dat = df[df['class'] == j]

	#arr = np.random.normal(size=500)
#	arr = dat['loss'].to_numpy()
#	ci = norm(*norm.fit(arr)).interval(0.80)  # fit a normal distribution and get 95% c.i.
#	height, bins, patches = plt.hist(arr, alpha=0.3)
#	plt.fill_betweenx([0, height.max()], ci[0], ci[1], color='g', alpha=0.1)  # Mark between 0 and the highest bar in the histogram
#	plt.show()

	outliers = dat.loc[dat['loss'] > ci[1]]

	if len(outliers) > 0:
		os.makedirs(os.path.join('/local/scratch/jrs596/dat/scrap_imgs', j), exist_ok = True)

	name = 0
	for idx, i in enumerate(dataset.imgs):
		if idx in outliers['IDX']:
			name += 1
			source = i[0]
			position = val_list.index(i[1])
			class_ = key_list[position]
			dest = os.path.join('/local/scratch/jrs596/dat/scrap_imgs', j, class_ + str(name) + '.jpg')
			shutil.copy(source, dest)



