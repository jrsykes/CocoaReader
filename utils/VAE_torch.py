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

import socket
from mpi4py import MPI

model_name = 'VAE'
"""
Initialize Hyperparameters
"""
batch_size = 2
learning_rate = 1e-3
num_epochs = 10
input_size = 124 #max input size for GPU 0 is 122, max input size for GPU 1 is 124. i.e. 2,349MiB / 24,576MiB
imgChannels = 3
n_filters = 5
imsize2 = input_size - (n_filters-1) * 2
convdim1 = 16
convdim2 = 32
zDim = 156


"""
Create dataloaders
"""

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dir = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised'
train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)



"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=imgChannels, featureDim=imgChannels*convdim2*input_size*input_size, zDim=zDim):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        
        self.encConv1 = nn.Conv2d(imgChannels, convdim1, n_filters)
        self.encConv2 = nn.Conv2d(convdim1, convdim2, n_filters)
        n_nodes = batch_size*convdim2*imsize2*imsize2
        self.encFC1 = nn.Linear(n_nodes, zDim)
        self.encFC2 = nn.Linear(n_nodes, zDim) 

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, n_nodes)
        self.decConv1 = nn.ConvTranspose2d(convdim2, convdim1, n_filters)
        self.decConv2 = nn.ConvTranspose2d(convdim1, imgChannels, n_filters)



    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x)) 
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
        x = F.relu(self.decFC1(z))
        x = x.view(batch_size, convdim2, imsize2 ,imsize2)
        x = F.relu(self.decConv1(x))
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
Allow model to train on all GPUs 
"""


host = socket.gethostname()
address = socket.gethostbyname(host)

comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()
info = dict()
info = comm.bcast(info, root=0)
info.update(dict(MASTER_ADDR=address, MASTER_PORT='1234'))
os.environ.update(info)

device = torch.device(f"cuda:{0}")

model.to(device)
model.train()

torch.distributed.init_process_group(backend='nccl', rank=rank, 
    world_size=world_size, store=None)

model = nn.parallel.DistributedDataParallel(model, device_ids=[device])


"""
Training 
"""
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
import sys
for epoch in range(num_epochs):
    for idx, data in enumerate(train_loader):
        imgs = data[0]
        imgs = imgs.to(device)
        # Feeding a batch of images into the network to obtain the output image, mu, and logVar

        out, mu, logVar = model(imgs)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_wts = copy.deepcopy(model.state_dict())

    # Save only the model weights for easy loading into a new model
    PATH = '/local/scratch/jrs596/VAE/models'
        
    final_out = {
        'model': model_wts,
        '__author__': 'Jamie R. Sykes'                    
        }    
                 
    model_path = os.path.join(PATH, model_name + '.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_out, f)

    # Save the whole model with pytorch save function
    torch.save(model.state_dict(), os.path.join(PATH, model_name + '.pth'))

    print('Epoch {}: Loss {}'.format(epoch, loss))



