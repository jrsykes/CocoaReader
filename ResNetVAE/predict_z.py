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
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
import time
from matplotlib import pyplot as plt
import numpy as np
from progress.bar import Bar

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 1000     # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability


# training parameters
epochs = 400        # training epochs
batch_size = 37
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# save model
save_model_path = '/local/scratch/jrs596/ResNetVAE/ForesArabData_Random_PredictedZ'

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)



def loss_function(recon_x, x, mu, logvar):
    #BCE = F.mse_loss(recon_x, x, reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def validation(model, device, test_loader):
    # set model as testing mode
    model.eval()
    n = len(test_loader.dataset)
    all_y, all_z = [], []
    with torch.no_grad():
        with Bar('Learning...', max=n/37) as bar:
            for X, y in test_loader:
                # distribute data to device
                X, y = X.to(device), y.to(device).view(-1, )
                X_reconst, z, mu, logvar = model(X) 

                all_y.extend(y.data.cpu().numpy())
                all_z.extend(z.data.cpu().numpy())
                bar.next()


    all_y = np.stack(all_y, axis=0)
    all_z = np.stack(all_z, axis=0)


    return all_y, all_z

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU



val_transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor()])


dir_ = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_VAE_filtered_unsplit'
#dir_ = '/local/scratch/jrs596/dat/test2/images'
dataset = torchvision.datasets.ImageFolder(dir_, transform=val_transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

# Create model
resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, 
    CNN_embed_dim=CNN_embed_dim, img_size=res_size).to(device)

#resnet_vae = nn.DataParallel(resnet_vae)

# record training process
epoch_train_losses = []
epoch_test_losses = []
check_mkdir(save_model_path)

y_test, z_test = validation(resnet_vae, device, loader)

np.save(os.path.join(save_model_path, 'y_cifar10_train_epoch.npy'), y_test)
np.save(os.path.join(save_model_path, 'z_cifar10_train_epoch.npy'), z_test)

