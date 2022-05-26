
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
from scipy.stats import norm
import shutil



# use same ResNet Encoder saved earlier!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 1000
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability

epoch = 5

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

from torchvision.utils import make_grid

original_list = []
recon_list_R = []
recon_list_IN = []
def validation(device, test_loader, indices, Class):
    # set model as testing mode
    #losses = pd.DataFrame(columns=['IDX', 'Class', 'R', 'IN'])
    losses = pd.DataFrame(columns=['IDX', 'Class', 'R'])
    losses['IDX'] = indices
    losses['Class'] = [Class] * len(indices)
    #paths = {'R':'/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_RandomWeights', 'IN':'/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_ImageNetWeights'}
    paths = {'R':'/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_RandomWeights'}

    for key, value in paths.items():
        loss = []
        model = load_net(value)
        model.eval()

        with torch.no_grad():
            for X, y in test_loader:

                X, y = X.to(device), y.to(device).view(-1, )
                X_reconst, z, mu, logvar = model(X)

                BCE = F.binary_cross_entropy(X_reconst, X, reduction='sum')
                loss.append(BCE.item())

                print(BCE)
                plt.imshow(X[0].to('cpu').permute(1,2,0))
                plt.axis('off')
                plt.show()
                plt.imshow(X_reconst[0].to('cpu').permute(1,2,0))
                plt.axis('off')
                plt.show()
                

        losses[key] = loss

#            print('Loss: ' + str(BCE))

#            X_reconstIN, z2, mu2, logvar2 = model_IN(X)
#            BCEIN = F.binary_cross_entropy(X_reconstIN, X, reduction='sum')
#            lossIN.append(BCEIN)
            
#            im = X_reconstR[0]#.cpu()
#            recon_list_R.append(im.permute(1, 2, 0))#
#            im = X_reconstIN[0]#.cpu()
#            recon_list_IN.append(im.permute(1, 2, 0))
#            im = X[0]#.cpu()
#            original_list.append(im.permute(1, 2, 0))
#          #

#        f, axarr =  plt.subplots(8, 3, sharex=True, figsize=(8,20))#

#        for index, value in enumerate(recon_list_R):
#            
#            axarr[index,2].imshow(recon_list_R[index])
#            axarr[index,2].set_title(str(round(lossR[index].item(),0)))
#            axarr[index,2].axis('off')
#            axarr[index,1].imshow(recon_list_IN[index])
#            axarr[index,1].set_title(str(round(lossIN[index].item(),0)))
#            axarr[index,1].axis('off')
#            axarr[index,0].imshow(original_list[index])
#            axarr[index,0].axis('off')
#        
#        axarr[0,0].text(0,0,'Original')
#        axarr[0,1].text(0,0, 'Image Net')
#        axarr[0,2].text(0,0, 'Random')
#        plt.show()


    return losses


def load_net(saved_model_path):
    # reload ResNetVAE model and weights
    resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, 
        CNN_embed_dim=CNN_embed_dim, img_size=res_size)
    weights = torch.load(os.path.join(saved_model_path, 'model_epoch{}.pth'.format(epoch)))#, map_location=torch.device('cpu'))
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


print('ResNetVAE epoch {} model reloaded!'.format(epoch))


val_transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor()])


#val_dir = '/local/scratch/jrs596/dat/PlantNotPlant_TinyIM+VAE_Filtered'
val_dir = '/local/scratch/jrs596/dat/VAE_test'

val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)


df = pd.DataFrame(columns=['IDX', 'Class', 'R'])

weights = 'R'

for key, value in val_dataset.class_to_idx.items():

    indices = []
    for i in val_dataset.imgs:
        if i[1] == int(value):
            indices.append(val_dataset.imgs.index(i))
    start = min(indices)
    stop = max(indices)
    class_ = Subset(val_dataset, range(start,stop+1))
    valid_loader = torch.utils.data.DataLoader(class_, num_workers=6, drop_last=False)

    losses = validation(device, valid_loader, indices, key)


#    arr = losses[weights].to_numpy()
#    ci = norm(*norm.fit(arr)).interval(0.90)#
#

#    outliers_up = losses.loc[losses[weights] > ci[1]]
#    outliers_dwn = losses.loc[losses[weights] < ci[0]]
#    outliers = pd.concat([outliers_up, outliers_dwn])#
#

#    name = 0#

#    for i in outliers['IDX'].tolist():
#        source = val_dataset.imgs[i][0]
#        #dest = os.path.join('/local/scratch/jrs596/dat/PNP_filtered_scrap_imgs', weights, key, key + str(name) + '.jpg')
#        #shutil.copy(source, dest)
#        os.remove(source)
#        name += 1#

#    
#    del outliers
#    print(key, 'done!')#




