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
from modules_tdist import *
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
import time
#from matplotlib import pyplot as plt
import numpy as np
import math
import scipy.stats

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 1000     # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability


# training parameters
epochs = 400        # training epochs
batch_size = 42
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# save model
save_model_path = '/local/scratch/jrs596/ResNetVAE/results_t-dist2'

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def plot_dist(z, t, epoch):
        #figure, axis = plt.subplots(2)
        plt.hist(z.cpu().detach().numpy())
        plt.title("Gausian")
        plt.grid(axis='y')
        plt.savefig(os.path.join(save_model_path, 'plots', str(epoch) + '_gausian.png'), format='png', dpi=200)

        plt.cla()
        plt.hist(t.cpu().detach().numpy())
        plt.title("Students t")
        plt.grid(axis='y')        
        plt.savefig(os.path.join(save_model_path, 'plots', str(epoch) + '_students_t.png'), format='png', dpi=200)


def loss_function(recon_x, x, t):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum').item()

    x_sample = torch.empty((batch_size, CNN_embed_dim))
    x_flat = x.view(42, -1)
    
    for i in range(batch_size):
        perm = torch.randperm(x_flat[i].size(0))
        idx = perm[:CNN_embed_dim]
        x_sample[i] = x_flat[i][idx]

    KLD = nn.KLDivLoss(reduction = "batchmean")
    KLD_loss = abs(KLD(x_sample.to('cuda'),t))


    #Add binary cross entropy loss between original and predicted image to the KL divergence fo the batch
    loss = (math.log(BCE)+torch.log(KLD_loss)).clone().detach().requires_grad_(True)

    return loss



def train(log_interval, model, device, dataloader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    all_y, all_z, all_mu, all_var = [], [], [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(dataloader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        optimizer.zero_grad()
        X_reconst, z, mu, var, t = model(X)  # VAE

        #plot_dist(z, t, epoch)
        
        loss = loss_function(X_reconst, X, t)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        all_y.extend(y.data.cpu().numpy())
        all_z.extend(z.data.cpu().numpy())
        all_mu.extend(mu.data.cpu().numpy())
        all_var.extend(var.data.cpu().numpy())

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(dataloader.dataset), 100. * (batch_idx + 1) / len(dataloader), loss.item()))
        

    all_y = np.stack(all_y, axis=0)
    all_z = np.stack(all_z, axis=0)
    all_mu = np.stack(all_mu, axis=0)
    all_var = np.stack(all_var, axis=0)

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_var, losses



# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
#device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
#use_cuda = False
device = torch.device("cuda")   # use CPU or GPU


# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if use_cuda else {}

train_transform = transforms.Compose([transforms.Resize([int(res_size*1.15), int(res_size*1.15)]),
                                transforms.RandomCrop(size=res_size),
                                transforms.RandomHorizontalFlip(),
                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.ToTensor()])


#train_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean/train'
train_dir = '/local/scratch/jrs596/dat/ResNetFung50+_images_organised/train'
#train_dir = '/scratch/staff/jrs596/dat/ResNetFung50+_images_organised/train'

#train_dir = '/local/scratch/jrs596/dat/test3'
#train_dir = '/scratch/staff/jrs596/dat/test3'
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)


#n = len(train_loader.dataset)


# Create model
resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, 
    CNN_embed_dim=CNN_embed_dim, img_size=res_size, batch_size=batch_size).to(device)

#resnet_vae = nn.DataParallel(resnet_vae)

model_params = list(resnet_vae.parameters())
optimizer = torch.optim.Adam(model_params, lr=learning_rate)


# record training process
epoch_train_losses = []
epoch_test_losses = []
check_mkdir(save_model_path)

writer = SummaryWriter(log_dir=os.path.join(save_model_path, 'logs'))


# start training
for epoch in range(epochs):
    # train, test model
    X_reconst_train, y_train, z_train, mu_train, var_train, train_losses = train(log_interval, resnet_vae, device, train_loader, optimizer, epoch)

    writer.add_scalar("Loss/train", torch.mean(torch.tensor(train_losses)), epoch)

    np.save(os.path.join(save_model_path, 'y_cifar10_train_epoch{}.npy'.format(epoch + 1)), y_train)
    np.save(os.path.join(save_model_path, 'z_cifar10_train_epoch{}.npy'.format(epoch + 1)), z_train)#



writer.close()
