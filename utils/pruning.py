from torchvision import models
import torch.nn as nn
import torch

import os
import pickle
import copy

import torch.nn.utils.prune as prune
import torch.nn.functional as F


#Initialize and Reshape the Networks
def initialize_model(num_classes,):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

# Initialize the model for this run
model = initialize_model(num_classes=2)


model_name = 'Plant-NotPlant_BinaryClasifier_FullDataSet_ResNet18'
pretrained_model_path = os.path.join('/local/scratch/jrs596/PNP_models/original', model_name + '.pkl')
pretrained_model_wts = pickle.load(open(pretrained_model_path, "rb"))

weights = copy.deepcopy(pretrained_model_wts['model'])

###############
#Remove 'module.' from layer names
new_keys = []
for key, value in weights.items():
    new_keys.append(key.replace('module.', ''))
for i in new_keys:
    weights[i] = weights.pop('module.' + i)#    

##############

model.load_state_dict(weights)

model.eval()
device = torch.device("cuda")
model.to(device)

parameters_to_prune = []


#for name, module in model.named_modules():
#	parameters_to_prune.append((model.conv1, 'weight'))#
#

#parameters_to_prune = tuple(parameters_to_prune)


parameters_to_prune = (
    (model.layer1.0.conv1, 'weight')
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

