
# coding: utf-8

# In[ ]:


#get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from modules import *
#from modules import ResNet_VAE
from sklearn.model_selection import train_test_split
import pickle
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import pandas as pd
import time

saved_model_path = '/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_356_LatentDim'

exp = 'cifar10'

# use same ResNet Encoder saved earlier!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 1000
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability

epoch = 12

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
#device = torch.device("cpu")


from torchvision.utils import make_grid


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    model.eval()
    loss = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            X_reconst, z, mu, logvar = model(X)
            im = X_reconst[0].cpu()
            im1 = im.permute(1, 2, 0)
            im = X[0].cpu()
            im2 = im.permute(1, 2, 0)
            
            f, axarr =  plt.subplots(2, sharex=True)
            axarr[0].imshow(im1)
            axarr[1].imshow(im2)
            
            plt.show()
            time.sleep(3)
 
            BCE = F.binary_cross_entropy(X_reconst, X, reduction='sum')
            loss.append(BCE)
    return loss



# reload ResNetVAE model and weights
resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, 
    CNN_embed_dim=CNN_embed_dim, img_size=res_size)
weights = torch.load(os.path.join(saved_model_path, 'model_epoch{}.pth'.format(epoch)))#, map_location=torch.device('cpu'))
resnet_vae.to(device)

model_params = list(resnet_vae.parameters())
optimizer = torch.optim.Adam(model_params)

#Remove 'module.' from layer names
new_keys = []
for key, value in weights.items():
    new_keys.append(key.replace('module.', ''))
for i in new_keys:
    weights[i] = weights.pop('module.' + i)#

resnet_vae.load_state_dict(weights)

print('ResNetVAE epoch {} model reloaded!'.format(epoch))


val_transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.ToTensor()])


val_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean/'

val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
#classes = val_dataset.classes

df = pd.DataFrame(columns=['IDX', 'class', 'loss'])


for key, value in val_dataset.class_to_idx.items():
    indices = []
    for i in val_dataset.imgs:
        if i[1] == int(value):
            indices.append(val_dataset.imgs.index(i))
    start = min(indices)
    stop = max(indices)
    class_ = Subset(val_dataset, range(start,stop))
    valid_loader = torch.utils.data.DataLoader(class_, num_workers=6)

    loss = validation(resnet_vae, device, optimizer, valid_loader)
    for idx, j in enumerate(loss):
        df.loc[len(df)] = [indices[idx], key, float(j)]
    print(key, 'done!')

df.to_csv('/local/scratch/jrs596/ResNetVAE/results/losses.csv', index=False)



