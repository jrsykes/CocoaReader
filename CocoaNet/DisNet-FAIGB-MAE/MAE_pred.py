

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

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
        'dim_3': 85, #Hard coded as 85, not 83, for now as this number needs to be divisible by number of attention heads
        'input_size': 240, #Hard coded as 240, not 233, for now this number needs to match the output size of the decoder
        'kernel_1': 3,
        'kernel_2': 5,
        'kernel_3': 13,
        'learning_rate': 0.00027319079821934975,
        'num_blocks_1': 4,
        'num_blocks_2': 1,
        'out_channels': int(107*1.396007582340178),
    }
        
    model = toolbox.build_model(num_classes=None, arch='DisNet_MaskedAutoencoder', config=config).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Load an image
def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).to(device)

# Visualize original and reconstructed images
def visualize(original, reconstructed):
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze(0).permute(1, 2, 0).cpu())
    plt.title('Original Image')
    
    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.permute(1, 2, 0).cpu())
    plt.title('Reconstructed Image')
    
    plt.show()
    plt.savefig('MAE_out_V2.png')

def main():
    # Define transformations for the test image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Assuming the input size is 128x128
        transforms.ToTensor()
    ])
    
    # Load the model and image
    model_path = '/local/scratch/jrs596/models/DisNet-FAIGB-MAE_pretty-sweep-16_weights.pth'
    image_path = '/local/scratch/jrs596/dat/FAIGB/FAIGB_FinalSplit_700/test/CocoaHealthy/1647617506.134829.jpeg'
    model = load_model(model_path)
    image = load_image(image_path, transform)
    
    mask = toolbox.generate_random_mask(image.unsqueeze(0).size(), device=image.device)


    with torch.no_grad():
        reconstructed = model(image.unsqueeze(0), mask)
    
    image = image * mask
    # Visualize the results
    visualize(image, reconstructed.squeeze(0))

if __name__ == '__main__':
    main()
