
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import random

dir_ = '/local/scratch/jrs596/dat/test_cocoa_imgen'

def img_reshape(img):
    img = Image.open(os.path.join(dir_,img)).convert('RGB')
    img = img.resize((300,300))
    img = np.asarray(img)
    return img



images = os.listdir(dir_)
images = random.sample(images, 400)

img_arr = []

for image in images:
    img_arr.append(img_reshape(image))

#pil_im = img_reshape('WheatTriticum113.jpeg')
#plt.imshow(np.asarray(pil_im))

rows=20
cols = 20


fig = plt.figure(figsize=(20., 20.))

grid = ImageGrid(fig, 111, 
                 nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes
                 )

for ax, im in zip(grid, img_arr):
	ax.axis('off')
	ax.imshow(im)

plt.show()