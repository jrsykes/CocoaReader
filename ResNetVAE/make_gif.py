
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
from modules_tdist import *
#from modules import ResNet_VAE
from sklearn.model_selection import train_test_split
import pickle
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import pandas as pd
import time
from scipy.stats import norm
import shutil



# use same ResNet Encoder saved earlier!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 1000
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability
batch_size = 1
epoch = list(range(28,31))


use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
#device = torch.device("cpu")


original_list = []
recon_list_R = []
recon_list_IN = []
def validation(device, test_loader, model, name):

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1, )
            
            X_reconst, z, mu, var, t = model(X)
            BCE = F.binary_cross_entropy(X_reconst[0], X.view(3, 224, 224), reduction='sum')
            print(BCE)

            plt.imshow(X_reconst[0].to('cpu').permute(1,2,0))
            #plt.imshow(X.view(3, 224, 224).to('cpu').permute(1,2,0))
            plt.axis('off')
            #plt.show()
            plt.savefig(os.path.join('/local/scratch/jrs596/dat/gif_images/egg_plant', str(name) + '.jpg'))


def load_net(saved_model_path, epoch):
    # reload ResNetVAE model and weights
    resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, 
        CNN_embed_dim=CNN_embed_dim, img_size=res_size, batch_size=batch_size)
    weights = torch.load(os.path.join(saved_model_path, 'model_epoch{}.pth'.format(epoch)))#, map_location=torch.device('cpu'))
    resnet_vae.to(device)   

#    new_keys = []
#    for key, value in weights.items():
#        new_keys.append(key.replace('module.', ''))
#    for i in new_keys:
#        weights[i] = weights.pop('module.' + i)#    

    resnet_vae.load_state_dict(weights)
    return resnet_vae


print('ResNetVAE epoch {} model reloaded!'.format(epoch))


val_transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor()])


val_dir = '/local/scratch/jrs596/dat/VAE_test_single'

val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)


#for key, value in val_dataset.class_to_idx.items():
for i in epoch:
    model = load_net('/local/scratch/jrs596/ResNetVAE/results_t-dist2', epoch=i)
    model.eval()

    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=6)#, drop_last=False)

    validation(device, valid_loader, model, name=i)



