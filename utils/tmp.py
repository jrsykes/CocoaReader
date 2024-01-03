#%%
import toolbox
import torch
from collections import OrderedDict
from torchvision import transforms, datasets
import os

device = 'cuda:0'

weights_path = '/users/jrs596/scratch/models/PhytNet_SR_FAIGB1.pth'

config = {
	'beta1': 0.9650025364732508,  
	'beta2': 0.981605256508036,  
	'dim_1': 79,  
	'dim_2': 107,  
	'dim_3': 93, #93 For PhytNetV0, 512 for ResNet18
	'input_size': 536,  
	'kernel_1': 5,  
	'kernel_2': 1,  
	'kernel_3': 7,  
	'learning_rate': 0.0002975957026209971,  
	'num_blocks_1': 2,  
	'num_blocks_2': 1,  
	'out_channels': 6,  
	'num_heads': 3, #3 for PhytNetV0, 4 for ResNet18  
	'batch_size': 42,  
	'num_decoder_layers': 4,
}
	
arch = 'PhytNet_SRAutoencoder'
model = toolbox.build_model(num_classes=config['out_channels'], arch=arch, config=config)


PhyloNetWeights = torch.load(weights_path, map_location=device)
PhyloNetWeights = {k: v for k, v in PhyloNetWeights.items() if 'downsample1' not in k}
PhyloNetWeights = {k: v for k, v in PhyloNetWeights.items() if 'downsample2' not in k}
PhyloNetWeights = {k: v for k, v in PhyloNetWeights.items() if 'poly' not in k}
# PhyloNetWeights = {k: v for k, v in PhyloNetWeights.items() if 'self_attn' not in k}
# PhyloNetWeights = {k: v for k, v in PhyloNetWeights.items() if 'norm' not in k}


new_state_dict = OrderedDict()
for key, value in PhyloNetWeights.items():
	new_key = key.replace('_orig_mod.', '')  # Remove the specific substring
	new_state_dict[new_key] = value

#strict = False allows for loading of a model with fc layer altered
model.load_state_dict(new_state_dict, strict=False)
model = model.to(device)
# %%

data_dir = '/users/jrs596/scratch/dat/sample_SR_images'
# Define the transformation to apply to the images
transform = transforms.Compose([
	transforms.ToTensor()  # Convert the images to tensors
])
# Create the dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
sample_dataloader = torch.utils.data.DataLoader(dataset, batch_size=9, shuffle=False)
	
# %%
import torchvision.utils as vutils

PATH = '/users/jrs596/tmp_SR_out'
os.makedirs(PATH, exist_ok=True)
			
# Assuming the image tensor is in the range [-1, 1]
def save_image(image, idx, path):
	# Convert back to [0, 255] and to 8-bit integer
	image = image * 255.0
	image = torch.clamp(image, 0, 255)
	image = image.byte()


	# Convert to PIL and save
	img_pil = transforms.ToPILImage()(image.cpu().detach())
	img_pil.save(os.path.join(path, f"image_{idx}.jpeg"))			

print("\nSaving sample images")    
for idx2, (image, _) in enumerate(sample_dataloader):                            
	_, SRdecoded = model(image.to(device))
	
	# #Method one
	# for img in SRdecoded:
	# 	img = img.squeeze(0).cpu().detach()
	# 	img = transforms.ToPILImage()(img)
	# 	img.save(os.path.join(PATH, "image_" + str(idx2) + ".jpeg"))
	
	# #Method two
	# grid = vutils.make_grid(SRdecoded, nrow=3, padding=0, normalize=False)
	# os.makedirs(PATH, exist_ok=True)
	# vutils.save_image(grid, os.path.join(PATH, "grid.png")) 
 
	for idx, img in enumerate(SRdecoded):
		#Crop one picel from right and bottom of image
		img = img[:, :-10, :-10]
	 
		img = img.squeeze(0)
		image = img * 255.0
		image = torch.clamp(image, 0, 255)
		image = image.byte()


		# Convert to PIL and save
		img_pil = transforms.ToPILImage()(image.cpu().detach())
		img_pil.save(os.path.join(PATH, f"image_{idx}.jpeg"))
  
		# save_image(img, idx2 * len(SRdecoded) + idx, PATH)
# %%
