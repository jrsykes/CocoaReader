#%%
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Load the model
model = torch.load('/local/scratch/jrs596/dat/models/efficientnet.pth')
model.eval()

# Check if CUDA is available and if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Load the image
image_path = '/local/scratch/jrs596/dat/IR_RGB_Comp_data/RGB_split_400/val/BPR/F4140024.JPG'  # replace with your image path
image = Image.open(image_path)

#%%

print(model)

#%%