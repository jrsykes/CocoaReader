import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
import copy
import pickle
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random


batch_size = 2#2
learning_rate = 1e-4
num_epochs = 50
input_size = 20#82
imgChannels = 1#3
n_filters = 5
imsize2 = input_size - (n_filters-1) * 3
convdim1 = 8
convdim2 = 16
convdim3 = 32
zDim = 300

"""
The following part takes a random image from test loader to feed into the VAE.
Both the original image and generated image from the distribution are shown.
"""
data_transforms = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor()])


test_dir = '/local/scratch/jrs596/dat/test'
#test_dir = '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised_test'
#test_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean_unorganised'
#test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

mnist_data = torchvision.datasets.MNIST('/local/scratch/jrs596/dat/mnist',train=False, transform=data_transforms)
test_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=4)

class VAE(nn.Module):
    def __init__(self, imgChannels=imgChannels, featureDim=int((batch_size*convdim3*imsize2*imsize2)/4), 
        zDim=zDim):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        layer2=int(featureDim/10)
        self.encConv1 = nn.Conv2d(imgChannels, convdim1, n_filters)
        self.encConv2 = nn.Conv2d(convdim1, convdim2, n_filters)
        self.encConv3 = nn.Conv2d(convdim2, convdim3, n_filters)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unPool = nn.MaxUnpool2d(2, stride=2)
        self.batchNorm = nn.BatchNorm2d(16)
        self.encFC1 = nn.Linear(featureDim, layer2)
        self.encFC1_1 = nn.Linear(layer2, layer2)
        self.encFC1_2 = nn.Linear(layer2, zDim)
        self.encFC2 = nn.Linear(featureDim, layer2)
        self.encFC2_1 = nn.Linear(layer2, layer2)
        self.encFC2_2 = nn.Linear(layer2, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1_1 = nn.Linear(zDim, layer2)
        self.decFC1_2 = nn.Linear(layer2, layer2)
        self.decFC1 = nn.Linear(layer2, featureDim)
        self.decConv1 = nn.ConvTranspose2d(convdim3, convdim2, n_filters)
        self.decConv2 = nn.ConvTranspose2d(convdim2, convdim1, n_filters)
        self.decConv3 = nn.ConvTranspose2d(convdim1, imgChannels, n_filters)

   

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        #x = self.pool(x) 
        x = F.relu(self.encConv2(x))
        #x = self.pool(x)
        x = self.batchNorm(x) 
        x = F.relu(self.encConv3(x))
        x, indices = self.pool(x)
        x_dim = np.prod(list(x.shape))
        x = x.view(-1, x_dim)
        mu = self.encFC1(x)
        mu = self.encFC1_1(mu)
        mu = self.encFC1_2(mu)
        logVar = self.encFC2(x)
        logVar = self.encFC2_1(logVar)
        logVar = self.encFC2_2(logVar)
        return mu, logVar, indices


    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z, indices):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1_1(z))
        x = F.relu(self.decFC1_2(x))
        x = F.relu(self.decFC1(x))
        x = x.view(batch_size, convdim3, int(imsize2/2) ,int(imsize2/2))
        x = self.unPool(x, indices, output_size=([batch_size,convdim3,imsize2,imsize2]))
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = torch.sigmoid(self.decConv3(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar, indices = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z, indices)
        return out, mu, logVar


device = torch.device('cpu')
model = VAE()
model.to(device)


model.load_state_dict(torch.load('/local/scratch/jrs596/VAE/models/VAE_MNIST.pth', map_location=torch.device('cpu')))
model.eval()



#with torch.no_grad():
#    for data in random.sample(list(test_loader), 10):
#        imgs, _ = data
#        imgs = imgs.to(device)
#        img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
#        plt.subplot(121)
#        plt.imshow(np.squeeze(img))
#        out, mu, logVAR = model(imgs)
#        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
#        plt.subplot(122)
#        plt.imshow(np.squeeze(outimg))
#        plt.show()

with torch.no_grad():
    for i in range(10):
        imgs, lables = next(iter(test_loader)) 
        
        imgs = imgs.to(device)
        img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
        plt.subplot(121)
        plt.imshow(np.squeeze(img))
        out, mu, logVAR = model(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))
        plt.show()