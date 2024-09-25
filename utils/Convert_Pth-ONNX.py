import sys
from torchvision import models, datasets, transforms
import torch
import os
import pickle
import numpy as np 
import torch.nn as nn
import datetime 

root = '/users/jrs596/scratch/models'
model_name = 'CocoaNet18_' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

# data_dir = "/local/scratch/jrs596/dat/split_cocoa_images"
data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca/Easy"

resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-SemiSupervised_NotCocoa_DFLoss2.pth"

ResNet18Weights = torch.load(resnet18_cococa_weights, map_location="cpu")

CocoaNet = models.resnet18(weights=None)
in_feat = CocoaNet.fc.in_features
CocoaNet.fc = nn.Linear(in_feat, 5)
CocoaNet.load_state_dict(ResNet18Weights, strict=True)
input_size = 375

CocoaNet.eval() 

# CocoaNet = torch.load(os.path.join(root, model_name + '.pth'), map_location=torch.device('cpu'))



# input_size = 750
# input_size = 750

from PIL import Image

image_file = "/users/jrs596/Monilia20.jpg"


def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)        
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHWll -h 
#    mean = np.array([0.079, 0.05, 0]) + 0.406
#    std = np.array([0.005, 0, 0.001]) + 0.224
#    for channel in range(image_data.shape[0]):
#        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255)
    image_data = np.expand_dims(image_data, 0)
    return image_data

x = preprocess_image(image_file, input_size, input_size)
x = torch.tensor(x)
#x = torch.randn(1, 3, input_size, input_size, requires_grad=True)


torch_out = CocoaNet(x)

# Export the model
torch.onnx.export(CocoaNet,              	# model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  os.path.join(root, model_name + '.onnx'),	# model input (or a tuple for multiple inputs)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names