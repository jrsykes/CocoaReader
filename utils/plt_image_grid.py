
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import math

dir_ = '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/FAIGB_combined_split/val/diseased'

def img_reshape(img):
    img = Image.open(os.path.join(dir_,img)).convert('RGB')
    img = img.resize((300,300))
    img = np.asarray(img)
    return img


n_imgs = 600#len(os.listdir(dir_))

#images = []
#for i in os.listdir(dir_):
#    dir_list = os.listdir(os.path.join(dir_,i))
#    for j in dir_list:
#        images.append(os.path.join(i,j))

images = os.listdir(dir_)

print(len(images))
images = random.sample(images, n_imgs)


img_arr = []

for image in images:
    img_arr.append(img_reshape(image))

rows = int(math.sqrt(n_imgs))

fig = plt.figure(figsize=(20., 20.))

grid = ImageGrid(fig, 111, 
                 nrows_ncols=(rows, rows),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes
                 )

for ax, im in zip(grid, img_arr):
	ax.axis('off')
	ax.imshow(im)

plt.show()
#plt.savefig("plt_grid.jpg")