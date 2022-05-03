
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

#saved_model_path = '/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_RandomWeights'

exp = 'cifar10'

# use same ResNet Encoder saved earlier!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 1000
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability

epoch = 400

use_cuda = torch.cuda.is_available()                   # check if GPU exists
#device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
device = torch.device("cpu")


from torchvision.utils import make_grid

original_list = []
recon_list_R = []
recon_list_IN = []
def validation(device, test_loader):
    # set model as testing mode
    
    lossR = []
    lossIN = []
    model_R = load_net('/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_RandomWeights')
    model_R.eval()
    model_IN = load_net('/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_ImageNetWeights')
    model_IN.eval()
    with torch.no_grad():
        for X, y in test_loader:

            X, y = X.to(device), y.to(device).view(-1, )
            X_reconstR, z, mu, logvar = model_R(X)

            BCER = F.binary_cross_entropy(X_reconstR, X, reduction='sum')
            lossR.append(BCER)
#            print('Loss: ' + str(BCE))

            X_reconstIN, z2, mu2, logvar2 = model_IN(X)
            BCEIN = F.binary_cross_entropy(X_reconstIN, X, reduction='sum')
            lossIN.append(BCEIN)
            
            im = X_reconstR[0].cpu()
            recon_list_R.append(im.permute(1, 2, 0))#
            im = X_reconstIN[0].cpu()
            recon_list_IN.append(im.permute(1, 2, 0))
            im = X[0].cpu()
            original_list.append(im.permute(1, 2, 0))
          

        f, axarr =  plt.subplots(8, 3, sharex=True, figsize=(8,20))

        for index, value in enumerate(recon_list_R):
            
            axarr[index,2].imshow(recon_list_R[index])
            axarr[index,2].set_title(str(round(lossR[index].item(),0)))
            axarr[index,2].axis('off')
            axarr[index,1].imshow(recon_list_IN[index])
            axarr[index,1].set_title(str(round(lossIN[index].item(),0)))
            axarr[index,1].axis('off')
            axarr[index,0].imshow(original_list[index])
            axarr[index,0].axis('off')
        
        axarr[0,0].text(0,0,'Original')
        axarr[0,1].text(0,0, 'Image Net')
        axarr[0,2].text(0,0, 'Random')
        plt.show()

 
            
    return loss








def load_net(saved_model_path):
    # reload ResNetVAE model and weights
    resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, 
        CNN_embed_dim=CNN_embed_dim, img_size=res_size)
    weights = torch.load(os.path.join(saved_model_path, 'model_epoch{}.pth'.format(epoch)), map_location=torch.device('cpu'))
    resnet_vae.to(device)   

    #model_params = list(resnet_vae.parameters())
    #optimizer = torch.optim.Adam(model_params)  

    #Remove 'module.' from layer names
    new_keys = []
    for key, value in weights.items():
        new_keys.append(key.replace('module.', ''))
    for i in new_keys:
        weights[i] = weights.pop('module.' + i)#    

    resnet_vae.load_state_dict(weights)
    return resnet_vae

#model_lst = []
#model_lst.append(load_net('/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_RandomWeights'))
#model_lst.append(load_net('/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_ImageNetWeights'))

print('ResNetVAE epoch {} model reloaded!'.format(epoch))


val_transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.ToTensor()])


#val_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean/'
val_dir = '/local/scratch/jrs596/dat/test'
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)

df = pd.DataFrame(columns=['IDX', 'class', 'loss'])



for key, value in val_dataset.class_to_idx.items():
    indices = []
    for i in val_dataset.imgs:
        if i[1] == int(value):
            indices.append(val_dataset.imgs.index(i))
    start = min(indices)
    stop = max(indices)
    class_ = Subset(val_dataset, range(start,stop+1))
    valid_loader = torch.utils.data.DataLoader(class_, num_workers=6, drop_last=False)

    loss = validation(device, valid_loader)
    for idx, j in enumerate(loss):
        df.loc[len(df)] = [indices[idx], key, float(j)]
    print(key, 'done!')

#df.to_csv('/local/scratch/jrs596/ResNetVAE/results/losses.csv', index=False)



