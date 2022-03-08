import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
from sklearn import metrics
import pandas as pd
import torchvision


## Convert onnx to tensorflow protobuf
import subprocess


bashCommand = "onnx-tf convert -i /local/scratch/jrs596/ResNetFung50_Torch/models/ResDes18_750dim.onnx -o /local/scratch/jrs596/ResNetFung50_Torch/models/ResDes18_750dim_TF"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

## Convert protobuf to TF-Lite
import tensorflow as tf
saved_model_dir = '/local/scratch/jrs596/ResNetFung50_Torch/models/ResDes18_750dim_TF'

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('/local/scratch/jrs596/ResNetFung50_Torch/models/TF_models/ResDes18_750dim.tflite', 'wb') as f:
  f.write(tflite_model)