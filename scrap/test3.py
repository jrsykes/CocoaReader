#%%

import torch

from transformers import DeformableDetrConfig, DeformableDetrModel

# Initializing a Deformable DETR SenseTime/deformable-detr style configuration
configuration = DeformableDetrConfig()

# Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
model = DeformableDetrModel(configuration)

# Accessing the model configuration
configuration = model.config

from thop import profile

def count_flops(model, input_size):
    inputs = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(inputs, ), verbose=False)

    # Convert to GFLOPs
    GFLOPs = flops / 1e9
    print(f'Total GFLOPs: {GFLOPs}')
    print(f'Total params: {params}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Deformable DETR model:\n\n")
# Assuming the input size is 1x3x800x800 for Deformable DETR
count_flops(model, (1, 3, 224, 224))

#%%

import torch
from torchvision.models import vit_b_16

# Initialize the model
vit = vit_b_16()

# If you want to use pre-trained weights, you can do so by passing the appropriate parameter:
vit = vit_b_16()

# If you want to use the model for inference, don't forget to set it to evaluation mode:
vit.eval()

print("VIT model:\n\n")
count_flops(vit, (1, 3, 224, 224))
# %%

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', _verbose=False)  # load silently

print("YOLO model:\n\n")
count_flops(yolo, (1, 3, 224, 224))
# %%
