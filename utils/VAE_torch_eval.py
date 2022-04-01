import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import random


batch_size = 1
#learning_rate = 1e-4
#num_epochs = 50
input_size = 177
imgChannels = 3
n_filters = 5
imsize2 = input_size - (n_filters-1) * 2
convdim1 = 16
convdim2 = 32
zDim = 180

"""
The following part takes a random image from test loader to feed into the VAE.
Both the original image and generated image from the distribution are shown.
"""
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ])
}

test_dir = '/home/jamiesykes/Downloads/ResNetFung50+_images_unorganised_test'
#test_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean_unorganised'
test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)



class VAE(nn.Module):
    def __init__(self, imgChannels=imgChannels, featureDim=batch_size*convdim2*imsize2*imsize2, zDim=zDim):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        
        self.encConv1 = nn.Conv2d(imgChannels, convdim1, n_filters)
        self.encConv2 = nn.Conv2d(convdim1, convdim2, n_filters)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
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

device = torch.device('cpu')
model = VAE()
model.to(device)


model.load_state_dict(torch.load('/home/jamiesykes/Downloads/VAE.pth'))
print('here')
exit(0)
model.eval()




#with torch.no_grad():
#    for data in random.sample(list(test_loader), 3):
#        img = data[0][-1:]
#        img = img.view(3,30,30)
#        imgplot = plt.imshow(img.permute(1, 2, 0))
#        plt.show()#

#        imgs = imgs.to(device)
#        out, mu, logVAR = model(imgs)
#        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
#        plt.subplot(122)
#        plt.imshow(np.squeeze(outimg))

with torch.no_grad():
    for data in random.sample(list(test_loader), 1):
        imgs, _ = data
        imgs = imgs.to(device)
        img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
        plt.subplot(121)
        plt.imshow(np.squeeze(img))
        out, mu, logVAR = model(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))