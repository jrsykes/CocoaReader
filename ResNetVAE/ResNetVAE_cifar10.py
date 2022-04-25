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


# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 1000     # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability


# training parameters
epochs = 200        # training epochs
batch_size = 20
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# save model
save_model_path = '/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_356_LatentDim'

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)



def loss_function(recon_x, x, mu, logvar):
    #BCE = F.mse_loss(recon_x, x, reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(log_interval, model, device, dataloader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    all_y, all_z, all_mu, all_logvar = [], [], [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(dataloader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        optimizer.zero_grad()
        X_reconst, z, mu, logvar = model(X)  # VAE
        
        loss = loss_function(X_reconst, X, mu, logvar)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        all_y.extend(y.data.cpu().numpy())
        all_z.extend(z.data.cpu().numpy())
        all_mu.extend(mu.data.cpu().numpy())
        all_logvar.extend(logvar.data.cpu().numpy())

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(dataloader.dataset), 100. * (batch_idx + 1) / len(dataloader), loss.item()))
        

    all_y = np.stack(all_y, axis=0)
    all_z = np.stack(all_z, axis=0)
    all_mu = np.stack(all_mu, axis=0)
    all_logvar = np.stack(all_logvar, axis=0)

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, losses


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y, all_z, all_mu, all_logvar = [], [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            X_reconst, z, mu, logvar = model(X)

            loss, BCE = loss_function(X_reconst, X, mu, logvar)
            test_loss += loss.item()  # sum up batch loss

            all_y.extend(y.data.cpu().numpy())
            all_z.extend(z.data.cpu().numpy())
            all_mu.extend(mu.data.cpu().numpy())
            all_logvar.extend(logvar.data.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    all_y = np.stack(all_y, axis=0)
    all_z = np.stack(all_z, axis=0)
    all_mu = np.stack(all_mu, axis=0)
    all_logvar = np.stack(all_logvar, axis=0)

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(test_loader.dataset), test_loss))
    return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, test_loss, BCE


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if use_cuda else {}

train_transform = transforms.Compose([transforms.Resize([int(res_size*1.15), int(res_size*1.15)]),
                                transforms.RandomCrop(size=res_size),
                                transforms.RandomHorizontalFlip(),
                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.ToTensor()])

#val_transform = transforms.Compose([transforms.Resize([448, 448]),
#                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                transforms.ToTensor()])


#train_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean/train'
train_dir = '/local/scratch/jrs596/dat/ResNetFung50+_images_organised/train'
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

#val_dir = '/local/scratch/jrs596/dat/ResNetFung50+_images_organised/val'
#val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
#valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=6)

# Create model
resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, 
    CNN_embed_dim=CNN_embed_dim, img_size=res_size).to(device)

resnet_vae = nn.DataParallel(resnet_vae)

print("Using", torch.cuda.device_count(), "GPU!")
model_params = list(resnet_vae.parameters())
optimizer = torch.optim.Adam(model_params, lr=learning_rate)
#optimizer = torch.optim.SGD(model_params, lr=0.001, momentum=0.9)


# record training process
epoch_train_losses = []
epoch_test_losses = []
check_mkdir(save_model_path)

writer = SummaryWriter(log_dir=os.path.join(save_model_path, 'logs'))


# start training
for epoch in range(epochs):
    # train, test model
    X_reconst_train, y_train, z_train, mu_train, logvar_train, train_losses = train(log_interval, resnet_vae, device, train_loader, optimizer, epoch)
    #X_reconst_test, y_test, z_test, mu_test, logvar_test, epoch_test_loss = validation(resnet_vae, device, optimizer, valid_loader)

    writer.add_scalar("Loss/train", mean(train_losses), epoch)
#    #writer.add_scalar("Loss/test", test_loss, epoch)#

    np.save(os.path.join(save_model_path, 'y_cifar10_train_epoch{}.npy'.format(epoch + 1)), y_train)
    np.save(os.path.join(save_model_path, 'z_cifar10_train_epoch{}.npy'.format(epoch + 1)), z_train)#

#    writer.flush()


writer.close()
