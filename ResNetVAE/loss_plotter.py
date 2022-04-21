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


img_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean/'

df = pd.read_csv('/local/scratch/jrs596/ResNetVAE/results/losses.csv', header=0)

classes = tuple(os.listdir(img_dir))

transform = transforms.Compose([transforms.Resize([224, 224]),
                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.ToTensor()])

dataset = torchvision.datasets.ImageFolder(img_dir, transform=transform)
#loader = torch.utils.data.DataLoader(dataset, num_workers=6)

#print(type(enumerate(dataset.imgs)))

indices = [22, 446, 345]

#for idx, i in enumerate(dataset.imgs):
#	if idx in indices:
#		print(i[0])



for i in classes:
	dat = df[df['class'] == i]

	#arr = np.random.normal(size=500)
	arr = dat['loss'].to_numpy()
	ci = norm(*norm.fit(arr)).interval(0.95)  # fit a normal distribution and get 95% c.i.
#	height, bins, patches = plt.hist(arr, alpha=0.3)
#	plt.fill_betweenx([0, height.max()], ci[0], ci[1], color='g', alpha=0.1)  # Mark between 0 and the highest bar in the histogram
#	plt.show()

	outliers = df.loc[df['loss'] < ci[0]]
	outliers.append(df.loc[df['loss'] > ci[1]], ignore_index=True)
	
	print(outliers)

	exit(0)
