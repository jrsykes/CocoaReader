#%%

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision import datasets, models, transforms
import math
from mpl_toolkits.axes_grid1 import ImageGrid

#%%

data_dir = '/local/scratch/jrs596/dat/tmp'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')



data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

image_dataset = datasets.ImageFolder(val_dir, data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=1)

images = []

for image, label in dataloader:
    images.append(image[0].permute(1,2,0))

    
# %%


def plt_grid():
    img_arr = []
    for image in images:
        img_arr.append(image)
    rows = int(math.sqrt(16))
    fig = plt.figure(figsize=(20., 20.))    

    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(rows, rows),  
                     axes_pad=0,  
                     )  

    for ax, im in zip(grid, img_arr):
        ax.axis('off')
        ax.imshow(im)  

    plt.show()
   

plt_grid()

# %%
