import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import random
import torch.nn.functional as F

sys.path.append(os.path.join(os.getcwd(), 'scripts/CocoaReader/utils'))
import toolbox
# Assuming the model classes are defined in the same script or imported
# If they are in a different file, make sure to import them
device = 'cuda:1'

# Load the model
def load_model(model_path):
    config = {
        'DFLoss_delta': 0.06379802231720144,
        'beta1': 0.928018334605748,
        'beta2': 0.943630021404608,
        'dim_1': 116,
        'dim_2': 106,
        'dim_3': 84, #Hard coded as 85, not 83, for now as this number needs to be divisible by number of attention heads
        'input_size': 240, #Hard coded as 240, not 233, for now this number needs to match the output size of the decoder
        'kernel_1': 3,
        'kernel_2': 5,
        'kernel_3': 13,
        'learning_rate': 0.00027319079821934975,
        'num_blocks_1': 4,
        'num_blocks_2': 1,
        'out_channels': int(107*1.396007582340178),
        'num_decoder_layers': 4,
        'num_heads': 3
    }
        
    model = toolbox.build_model(num_classes=None, arch='DisNet_SRAutoencoder', config=config).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Load an image
def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).to(device)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize(images, reconstructions):
    num_images = len(images)
    
    # Calculate the width and height ratios based on the image sizes
    total_width = 240 + 356
    total_height = max(240, 356)
    
    width_ratio_original = 240 / total_width
    width_ratio_reconstructed = 356 / total_width
    
    fig = plt.figure(figsize=(total_width/100, num_images * total_height/100))
    
    spec = gridspec.GridSpec(num_images, 2, width_ratios=[width_ratio_original, width_ratio_reconstructed], wspace=0.05)
    
    for i, (original, reconstructed) in enumerate(zip(images, reconstructions)):
        # Original image
        ax = fig.add_subplot(spec[i, 0])
        ax.imshow(original.squeeze(0).permute(1, 2, 0).cpu())
        ax.axis('off')
        
        # Reconstructed image
        ax = fig.add_subplot(spec[i, 1])
        ax.imshow(reconstructed.permute(1, 2, 0).cpu())
        ax.axis('off')
    
    plt.savefig('SAE_out_V1.png')





def main():
    # Define transformations for the test image
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.ToTensor()
    ])
    
    # Load the model
    model_path = '/scratch/staff/jrs596/models/FAIGB_MAE_subirrigation_75_weights.pth'
    model = load_model(model_path)
    
    base_dir = '/scratch/staff/jrs596/dat/FAIGB/FAIGB_FinalSplit_700_TrainVal/val/'
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    #shuffle subdirs
    random.shuffle(subdirs)
    subdirs = subdirs[0:10]
    
    images = []
    reconstructions = []
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        image_name = os.listdir(subdir_path)[0]  # take the first image
        image_path = os.path.join(subdir_path, image_name)
        
        SRimage = load_image(image_path, transform)

        image = F.interpolate(SRimage.unsqueeze(0), size=(240, 240), mode='bilinear', align_corners=True)

        with torch.no_grad():
            _, reconstructed = model(image)
        
        images.append(image)
        reconstructions.append(reconstructed.squeeze(0))
    
    # Visualize the results
    visualize(images, reconstructions)

main()
