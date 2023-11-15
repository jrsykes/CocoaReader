import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import random

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
        
    model = toolbox.build_model(num_classes=None, arch='DisNet_MaskedAutoencoder', config=config).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Load an image
def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).to(device)

# # Visualize original and reconstructed images
# def visualize(original, reconstructed):
#     plt.figure(figsize=(10, 5))
    
#     # Original image
#     plt.subplot(1, 2, 1)
#     plt.imshow(original.squeeze(0).permute(1, 2, 0).cpu())
#     plt.title('Original Image')
    
#     # Reconstructed image
#     plt.subplot(1, 2, 2)
#     plt.imshow(reconstructed.permute(1, 2, 0).cpu())
#     plt.title('Reconstructed Image')
    
#     plt.savefig('MAE_out_V2.png')

# def main():
#     # Define transformations for the test image
#     transform = transforms.Compose([
#         transforms.Resize((240, 240)),  # Assuming the input size is 128x128
#         transforms.ToTensor()
#     ])
    
#     # Load the model and image
#     model_path = '/scratch/staff/jrs596/models/FAIGB_MAE_vibrant-sweep-20_weights.pth'
#     image_path = '/scratch/staff/jrs596/dat/FAIGB/FAIGB_FinalSplit_700_TrainVal/val/CocoaHealthy/1647617506.134829.jpeg'
#     model = load_model(model_path)
#     image = load_image(image_path, transform)
    
#     mask = toolbox.generate_random_mask(image.unsqueeze(0).size(), device=image.device)


#     with torch.no_grad():
#         reconstructed = model(image.unsqueeze(0), mask)
    
#     image = image * mask
#     # Visualize the results
#     visualize(image, reconstructed.squeeze(0))

# if __name__ == '__main__':
#     main()


###############################################


# Visualize original and reconstructed images
def visualize(images, reconstructions):
    num_images = len(images)
    plt.figure(figsize=(10*num_images, 5))
    
    for i, (original, reconstructed) in enumerate(zip(images, reconstructions)):
        # Original image
        plt.subplot(2, num_images, i+1)
        plt.imshow(original.squeeze(0).permute(1, 2, 0).cpu())
        plt.title(f'Original Image {i+1}')
        
        # Reconstructed image
        plt.subplot(2, num_images, num_images+i+1)
        plt.imshow(reconstructed.permute(1, 2, 0).cpu())
        plt.title(f'Reconstructed Image {i+1}')
    
    plt.savefig('MAE_out_NoMaskV3.png')

def main():
    # Define transformations for the test image
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor()
    ])
    
    # Load the model
    model_path = '/scratch/staff/jrs596/models/FAIGB_MAE_vibrant-sweep-20_weights.pth'
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
        
        image = load_image(image_path, transform)
        mask = toolbox.generate_random_mask(image.unsqueeze(0).size(), device=image.device)
        mask = torch.ones_like(mask)


        with torch.no_grad():
            reconstructed = model(image.unsqueeze(0), mask)
        
        image = image * mask
        images.append(image)
        reconstructions.append(reconstructed.squeeze(0))
    
    # Visualize the results
    visualize(images, reconstructions)

main()
