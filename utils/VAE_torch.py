"""
Import necessary libraries to create a variational autoencoder
The code is mainly developed using the PyTorch library
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
import copy
import pickle
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

import socket
from mpi4py import MPI
import time

model_name = 'VAE'
"""
Initialize Hyperparameters
"""
batch_size = 4
learning_rate = 1e-4
num_epochs = 50
input_size = 177
imgChannels = 3
n_filters = 5
imsize2 = input_size - (n_filters-1) * 2
convdim1 = 16
convdim2 = 32
zDim = 180

writer = SummaryWriter(log_dir='/local/scratch/jrs596/VAE/logs')

"""
Create dataloaders
"""

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
}

train_dir = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised'
#train_dir = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised_test'
train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

running_mean = 0.0
running_var = 0.0
for idx, data in enumerate(train_loader):
    running_mean += data[0].mean() * batch_size
    running_var += data[0].var() * batch_size

n = len(train_loader.dataset)
mean = running_mean/n
var = running_var/n

if n%batch_size != 0:
    print('Total N samples must be divisable by batch size')
    print('N = ' + str(n))
    exit(0)


class VAE(nn.Module):
    def __init__(self, imgChannels=imgChannels, featureDim=batch_size*convdim2*imsize2*imsize2, zDim=zDim):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        
        self.encConv1 = nn.Conv2d(imgChannels, convdim1, n_filters).to('cuda:0')
        self.encConv2 = nn.Conv2d(convdim1, convdim2, n_filters).to('cuda:0')
        self.encFC1 = nn.Linear(featureDim, zDim).to('cuda:0')
        self.encFC2 = nn.Linear(featureDim, zDim).to('cuda:0')

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim).to('cuda:1')
        self.decConv1 = nn.ConvTranspose2d(convdim2, convdim1, n_filters).to('cuda:0')
        self.decConv2 = nn.ConvTranspose2d(convdim1, imgChannels, n_filters).to('cuda:0')
   

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x.to('cuda:0'))) 
        x = F.relu(self.encConv2(x))
        x_dim = np.prod(list(x.shape))
        x = x.view(-1, x_dim)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar


    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z.to('cuda:1')))
        x = x.view(batch_size, convdim2, imsize2 ,imsize2)
        x = F.relu(self.decConv1(x.to('cuda:0')))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


model = VAE()

"""
Training 
"""
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(num_epochs):
    running_ELBO = 0.0
    running_CE = 0.0
    running_KL = 0.0
    
    
    with Bar('Learning...', max=n/batch_size) as bar:
          for idx, data in enumerate(train_loader):
                    
            imgs = data[0]
            labels = imgs.to('cuda:0')
            # Feeding a batch of images into the network to obtain the output image, mu, and logVar 

            out, mu, logVar = model(imgs)   

            """
            My KL divergence definition
            """
            kl_divergence = torch.mean(-mean * (var.log() + logVar - mu**2 - logVar.exp()))
            
            binary_cross_entropy = F.binary_cross_entropy(out, labels)  
            loss = binary_cross_entropy + kl_divergence 

            running_ELBO += loss.item() * imgs.shape[0]
            running_CE += binary_cross_entropy.item() * imgs.shape[0]
            running_KL += kl_divergence.item() * imgs.shape[0]  

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  

            bar.next()
            
    epoch_loss = round(float(running_ELBO / n),4)
    epoch_CE = round(float(running_CE / n),4)
    epoch_KL = round(float(running_KL / n),4)

    writer.add_scalar("ELBO/train", epoch_loss, epoch)
    writer.add_scalar("Cross entropy/train", epoch_CE, epoch)
    writer.add_scalar("KL divergence/train", epoch_KL, epoch)

    print('Saving checkpoint')
    PATH = '/local/scratch/jrs596/VAE/models'
    torch.save(model.state_dict(), os.path.join(PATH, model_name + '.pth'))

    print('Epoch {}: ELBO loss {}'.format(epoch, epoch_loss))
    print('KL divergence {}: Binary Cross Entropy {}'.format(epoch_KL, epoch_CE))

writer.flush()
writer.close()

