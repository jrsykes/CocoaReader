import random
import os
import shutil

pth = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised/train/'
pth2 = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised_test/train/'
pth3 = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised_train/train/'

images = tuple(os.listdir('/local/scratch/jrs596/dat/ResNetFung50+_images_organised/train'))

print(images)
exit(0)

random.shuffle(images)

for i in images[0:100]:
	src = os.path.join(pth, i)
	dst = os.path.join(pth2, i)
	shutil.copy(src,dst)

images = os.listdir(pth)
random.shuffle(images)

for i in images[0:100]:
	src = os.path.join(pth, i)
	dst = os.path.join(pth3, i)
	shutil.copy(src,dst)