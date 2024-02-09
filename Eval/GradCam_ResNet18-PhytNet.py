#%%
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import torch.nn.functional as F
import os
from torchvision import datasets, transforms, models
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
from torchvision.transforms import ToPILImage
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
from ArchitectureZoo import PhytNetV0
from PIL import Image

device = torch.device("cuda:0")
# device = torch.device("cpu")

#%%
   
class ResNet_CAM(nn.Module):
	def __init__(self, net, layer_k=-1):
		super(ResNet_CAM, self).__init__()
		self.resnet = net
		convs = nn.Sequential(*list(net.children())[:-1])
		self.first_part_conv = convs[:layer_k]
		self.second_part_conv = convs[layer_k:]
		self.linear = nn.Sequential(*list(net.children())[-1:])
		
	def forward(self, x):
		x = self.first_part_conv(x.to(device))
		x.register_hook(self.activations_hook)
		x = self.second_part_conv(x)
		x = F.adaptive_avg_pool2d(x, (1,1))
		x = x.view((1, -1))
		x = self.linear(x)
		return x
	
	def activations_hook(self, grad):
		self.gradients = grad
	
	def get_activations_gradient(self):
		return self.gradients
	
	def get_activations(self, x):
		return self.first_part_conv(x.to(device))
  
class PhytNet_CAM(nn.Module):
	def __init__(self, net):
		super(PhytNet_CAM, self).__init__()
		self.phytnet = net

		# Extract relevant layers for CAM
		self.features = nn.Sequential(
			net.conv1,
			net.gn1,
			net.relu,
			net.maxpool,
			net.layer1,
			net.layer2
		)

		self.global_avg_pool = net.global_avg_pool
		self.fc = net.fc
		
	def forward(self, x):
		# Extract feature maps from the network
		x = self.features(x)
		x.register_hook(self.activations_hook)

		# Apply global average pooling and pass through the final classifier
		pooled = self.global_avg_pool(x)
		pooled = torch.flatten(pooled, 1)
		output = self.fc(pooled)

		return output

	def activations_hook(self, grad):
		self.gradients = grad

	def get_activations_gradient(self):
		return self.gradients

	def get_activations(self, x):
		return self.features(x)
	  
# %%

def superimpose_heatmap(heatmap, img):
	resized_heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))

	# Replace NaN or Inf with 0
	resized_heatmap = np.nan_to_num(resized_heatmap, nan=0, posinf=0, neginf=0)

	# Normalizing the heatmap between 0 and 1 for eliminating any invalid values
	min_val = np.min(resized_heatmap)
	max_val = np.max(resized_heatmap)
	if max_val > min_val:  # To avoid division by zero
		resized_heatmap = (resized_heatmap - min_val) / (max_val - min_val)
	else:
		resized_heatmap = np.zeros_like(resized_heatmap)

	resized_heatmap = np.uint8(255 * resized_heatmap)  # Now it can be correctly casted to uint8
	resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)

	# Converting image to the same data type as superimposed_img
	img_as_float = img[0].permute(1,2,0).cpu().numpy().astype(float)

	superimposed_img = cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB) * 0.006 + img_as_float
	superimposed_img = torch.from_numpy(superimposed_img)  # Convert back to tensor if necessary

	return superimposed_img



def get_grad_cam(net, img):
	net.eval()
	img = img.to(device)
	pred = net(img)
	pred[:,pred.argmax(dim=1)].backward()
	gradients = net.get_activations_gradient()
	pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
	activations = net.get_activations(img).detach()
	for i in range(activations.size(1)):
		activations[:, i, :, :] *= pooled_gradients[i]
	heatmap = torch.mean(activations, dim=1).squeeze().detach().to('cpu')
	heatmap = np.maximum(heatmap, 0)
	heatmap /= torch.max(heatmap)

	
	return superimpose_heatmap(heatmap, img)


out_labels = ["BPR", "FPR", "Healthy", "Not Cocoa", "WBD"]  

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 70)  # Replace with the path to a .ttf file on your system


ResNet18_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-SemiSupervised_NotCocoa_DFLoss.pth"

ResNet18Weights = torch.load(ResNet18_weights, map_location=device)
ResNet18 = models.resnet18(weights=None)
in_feat = ResNet18.fc.in_features
ResNet18.fc = nn.Linear(in_feat, 5)
ResNet18.load_state_dict(ResNet18Weights, strict=True)
ResNet18.eval()
ResNet18.to(device)
	


#cool-sweep-42
config = {
    'beta1': 0.9671538235629524,
    'beta2': 0.9574398373980104,
    'dim_1': 126,
    'dim_2': 91,
    'dim_3': 89,
    'input_size': 371,
    'kernel_1': 5,
    'kernel_2': 1,
    'kernel_3': 17,
    'learning_rate': 9.66816458944127e-05,
    'num_blocks_1': 2,
    'num_blocks_2': 1,
    'out_channels': 7
}


PhytNet= PhytNetV0(config=config).to(device)
# # Load weights
PhytNet_weights_path = "/users/jrs596/scratch/models/PhytNet-Cocoa-SemiSupervised_NotCocoa_183.pth"
weights = torch.load(PhytNet_weights_path, map_location=lambda storage, loc: storage.cuda(0))

PhytNet.load_state_dict(weights, strict=False)
PhytNet.eval()
PhytNet.to(device)

models_ = ["Input", PhytNet, ResNet18]


dat_dir = '/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca/Easy/val'
class_labels = os.listdir(dat_dir)
class_labels.sort()

def load_data(dat_dir):
	data_transforms = transforms.Compose([
		# transforms.Resize((img_size,img_size)),
		transforms.ToTensor()
	])

	valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
	valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=1)

	return valloader

#%%

img_size = 375
n_imgs = 40
n_models = 2

plot_img_size = img_size
valloader = load_data(dat_dir)


imgs = torch.Tensor(n_imgs, n_models, 3, plot_img_size, plot_img_size)  # Adjusted the tensor dimensions
imgs_np = imgs.numpy()
ground_truth_labels = []

fig, axs = plt.subplots(n_imgs, n_models, figsize=(100, 100))
fig.set_figwidth(40)
plt.subplots_adjust(wspace=0, hspace=0)
# Create a new blank image
grid_img = Image.new('RGB', (n_models * img_size, n_imgs * img_size))


	
#%%    


for i, (img, label) in enumerate(valloader):
	if i >= n_imgs:
		break


	
	for idx, model in enumerate(models_):
		if idx == 0:  # For the first column, use the ground truth label
			img_size = 371
			img_resized = F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)
			
			pil_img = Image.fromarray((img_resized.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
			draw = ImageDraw.Draw(pil_img)
			draw.text((0, 0), class_labels[label], fill="white", font=font)
			grid_img.paste(pil_img, (idx * img_size, i * img_size))
		elif idx == 1:  # For the second column, use the Grad-CAM output with predicted class
			# model = PhytNet
			model_cam_net = PhytNet_CAM(model)
			img_size = config['input_size']  
			img_resized = F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)
 
	  
			out = get_grad_cam(model_cam_net, img_resized)  
			#Normalise GradCAM output
			out = (out - out.min()) / (out.max() - out.min())
			out = out.permute(2,0,1).cpu().mul(255).byte()

			_, _, pred = model(img_resized.to(device))
			pred = pred.argmax(dim=1)

			# Convert out to PIL image and draw the predicted class
			out_pil_1 = ToPILImage()(out)
			out_pil_1 = out_pil_1.resize((375, 375))

			draw = ImageDraw.Draw(out_pil_1)
			draw.text((0, 0), out_labels[pred], fill="white", font=font)

			grid_img.paste(out_pil_1, (idx * img_size, i * img_size))
		elif idx == 2:  # For the second column, use the Grad-CAM output with predicted class
			# model = ResNet18
			model_cam_net = ResNet_CAM(model)
			img_size = 375
			img_resized = F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)
	
			out = get_grad_cam(model_cam_net, img_resized)  
			#Normalise GradCAM output
			out = (out - out.min()) / (out.max() - out.min())
			out = out.permute(2,0,1).cpu().mul(255).byte()

			pred = model(img_resized.to(device))
			pred = pred.argmax(dim=1)

			# Convert out to PIL image and draw the predicted class
			out_pil_2 = ToPILImage()(out)
			draw = ImageDraw.Draw(out_pil_2)
			draw.text((0, 0), out_labels[pred], fill="white", font=font)

			grid_img.paste(out_pil_2, (idx * img_size, i * img_size))
			
	# Create a new blank image with the combined dimensions
	new_img = Image.new('RGB', (img_size*len(models_), img_size))
	# Paste the images side by side
	new_img.paste(pil_img, (0, 0))
	new_img.paste(out_pil_1, (img_size, 0))
	new_img.paste(out_pil_2, (img_size*2, 0))

	# Save the image
	dir_ = "/users/jrs596/GradCAM_imgs/GradCAM_ResNet18_PhytNet138K_Easy"
	os.makedirs(dir_, exist_ok=True)
	new_img.save(os.path.join(dir_, str(i) + ".png"))

# grid_img.save("/users/jrs596/GradCAM_imgs/GradCAM_ResNet18_PhytNet_Difficult.png")

print("Done!")
