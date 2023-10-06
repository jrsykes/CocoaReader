#%%
import shutil
import os
from PIL import Image, ImageOps

root = "/home/jamiesykes/Documents/Ecuador_data"

sure = os.path.join(root, "EcuadorImages_EL_LowRes_17_03_23_sure")
unsure = os.path.join(root, "EcuadorImages_EL_LowRes_17_03_23")

#%%
sure_imgs = []
for d in os.listdir(sure):
    for sd in os.listdir(os.path.join(sure, d)):
        imgs = os.listdir(os.path.join(sure, d, sd))
        sure_imgs.append(imgs)
       
sure_imgs = [item for sublist in sure_imgs for item in sublist]

for d in os.listdir(unsure):
    for sd in os.listdir(os.path.join(unsure, d)):
        imgs = os.listdir(os.path.join(unsure, d, sd))
        
        for i in imgs:
            if i in sure_imgs:
                os.remove(os.path.join(unsure, d, sd, i))


#%%