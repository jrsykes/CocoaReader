import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
from sklearn import metrics
import pandas as pd
import torchvision


model_path = '/local/scratch/jrs596/ResNetFung50_Torch/models'
model = '/model.pth'
data_dir = "/local/scratch/jrs596/dat/ResNetFung50+_images_organised_subset"
#data_dir = "/local/scratch/jrs596/dat/compiled_cocoa_images/split"

model = torch.load(model_path + model)
#model.eval()

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")
model.to(device)


dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
input_names = os.listdir('/local/scratch/jrs596/dat/ResNetFung50+_images_organised_subset/val')
output_names = ['AauberginesDiseased']
torch.onnx.export(model.module, dummy_input, model_path + '/model.onnx', 
	verbose=True, input_names=input_names, output_names=output_names)


## Convert onnx to tensorflow protobuf
import subprocess


bashCommand = "onnx-tf convert -i /local/scratch/jrs596/ResNetFung50_Torch/models/model.onnx -o /local/scratch/jrs596/ResNetFung50_Torch/models/model_tensorflow"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

## Convert protobuf to TF-Lite
import tensorflow as tf
saved_model_dir = '/local/scratch/jrs596/ResNetFung50_Torch/models/model_tensorflow'
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('/local/scratch/jrs596/ResNetFung50_Torch/models/model.tflite', 'wb') as f:
  f.write(tflite_model)