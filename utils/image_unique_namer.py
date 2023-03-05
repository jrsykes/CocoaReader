#%%
import os
import time
import shutil

root ='/local/scratch/jrs596/dat/ElodeaProject/Elodea'


for i in os.listdir(root + '/Summer2022_Photos'):
    print(i)
    path1 = root + '/Summer2022_Photos/' + i + '/DCIM'
    for j in os.listdir(path1):
        images = os.listdir(path1 + '/' + j)
        
        for k in images:
            src = os.path.join(path1, j, k)
            dst = os.path.join(root, 'Combined_training_images', j + str(time.time()) + '.jpeg')

            shutil.copy(src, dst)


# %%
