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
CNN_embed_dim = 356     # latent dim extracted by 2D CNN
#res_size = 448        # ResNet image size
dropout_p = 0.2       # dropout probability


convnext_vae = ConvNeXt_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim)
