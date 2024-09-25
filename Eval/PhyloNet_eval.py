import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
import pandas as pd
import sys
import RobinsonFoulds


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('/users/jrs596/scripts/CocoaReader/utils')

data_dir = "/users/jrs596/scratch/dat/flowers102_split"

model_path = '/users/jrs596/scratch/models/ResNet18_RW_PhyloPT_Flowers_MSE+ESS.pth'
taxonomy = pd.read_csv(os.path.join('/users/jrs596/scratch/dat/flowers102_split/flowers_taxonomy.csv'), header=0)

model = models.resnet18(weights=None)

class ModifiedResNet18(nn.Module):
	def __init__(self, original_model):
		super(ModifiedResNet18, self).__init__()
		# Keep all layers except the last two (the fc layer)
		self.features = nn.Sequential(*list(original_model.children())[:-2])
		self.avgpool = original_model.avgpool
		# Initialize the fully connected layer with the same output size as the original
		# self.fc = original_model.fc
	def forward(self, x):
		# Pass input through the feature layers
		x = self.features(x)
		# Pass through the average pooling layer
		avgpool_output = self.avgpool(x)
		# Flatten the output before passing it to the fully connected layer
		# flat = torch.flatten(avgpool_output, 1)
		# Pass through the fully connected layer for standard output
		# fc_output = self.fc(flat)
		# Return both the standard output and the avgpool output
		return avgpool_output.squeeze()
	  
# Instantiate the modified model
model = ModifiedResNet18(model)

# Load the state dictionary
state_dict = torch.load(model_path)

# Modify the keys by removing '_orig_mod.'
new_state_dict = {}
for key, value in state_dict.items():
	new_key = key.replace("_orig_mod.", "")
	new_state_dict[new_key] = value

# Load the modified state dictionary into the model
model.load_state_dict(new_state_dict, strict=True)
model = model.to(device)

input_size = 293
batch_size = 60

data_transforms = {
	'train': transforms.Compose([
		transforms.Resize((input_size,input_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
	]),
	'val': transforms.Compose([
		transforms.Resize((input_size,input_size)),
		transforms.ToTensor(),
	])
	}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
	for x in ['train', 'val']}#, 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
	shuffle=True, num_workers=6) for x in ['train', 'val']}





for i, (inputs, labels) in enumerate(dataloaders_dict['val'],0 ):
	
	inputs = inputs.to(device)
	labels = labels.to(device)
	encoded_pooled = model(inputs)

	trees, _ = RobinsonFoulds.trees(taxonomy, labels, encoded_pooled)


	PATH = os.path.join('/users/jrs596/scratch/dat/flowers102_split/eval_trees')           
	os.makedirs(PATH, exist_ok=True)
	trees["pred_tree"].write(format=1, outfile=os.path.join(PATH, "tree_pred.newick"))
	trees["target_tree"].write(format=1, outfile=os.path.join(PATH, "tree_true.newick"))    

	break