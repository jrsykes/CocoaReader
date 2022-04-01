import random
import os
import shutil

pth = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised/train/'
pth2 = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised_test/train/'

images = os.listdir(pth)
random.shuffle(images)

for i in images[0:40]:
	src = os.path.join(pth, i)
	dst = os.path.join(pth2, i)
	shutil.copy(src,dst)