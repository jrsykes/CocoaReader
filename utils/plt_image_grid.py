
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import math

dir_ = '/local/scratch/jrs596/dat/FAIGB_test/train/diseased'
dir_ = '/home/jamiesykes/Downloads/diseased_HandFiltered'

n_imgs = 1#len(os.listdir(dir_))
images = os.listdir(dir_)
print(len(images))
#images = random.sample(images, n_imgs)  


def img_reshape(img):
    img = Image.open(os.path.join(dir_,img)).convert('RGB')
    img = img.resize((300,300))
    img = np.asarray(img)
    return img

im = img_reshape(images[0])

def plt_grid():
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



def plot_dist(im):
        #figure, axis = plt.subplots(2)
        im = im.reshape(1,im.size)
        plt.hist(im)
        plt.title("Single image")
        plt.grid(axis='y')
        plt.show()
        #plt.savefig(os.path.join(save_model_path, 'plots', str(epoch) + '_gausian.png'), format='png', dpi=200)

plot_dist(im)
