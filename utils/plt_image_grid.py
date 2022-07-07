
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import math

dir_ = 'Forestry_ArableImages_GoogleBing_PNP_out/NotPlant'

def img_reshape(img):
    img = Image.open(os.path.join(dir_,img)).convert('RGB')
    img = img.resize((300,300))
    img = np.asarray(img)
    return img


n_imgs = 200#len(os.listdir(dir_))


images = os.listdir(dir_)
images = random.sample(images, n_imgs)

img_arr = []

for image in images:
    img_arr.append(img_reshape(image))

#pil_im = img_reshape('WheatTriticum113.jpeg')
#plt.imshow(np.asarray(pil_im))

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