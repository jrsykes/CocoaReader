import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
from sklearn import metrics
import pandas as pd
import time
import pickle
import copy
from torch.utils.mobile_optimizer import optimize_for_mobile

root = '/local/scratch/jrs596/dat/models'
model_path = 'CocoaNet18_DN.pth'
data_dir = "/local/scratch/jrs596/dat/split_cocoa_images"

input_size = 750
batch_size = 37
criterion = nn.CrossEntropyLoss()

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ]),
}

# Create training and validation datasets
image_datasets = {'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])}
# Create training and validation dataloaders
dataloaders_dict = {'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, 
	shuffle=False, num_workers=1)}

num_classes = len(os.listdir(os.path.join(data_dir, 'val')))

for images, labels in dataloaders_dict['val']:
	input_fp32 = images
	break


def Remove_module_from_layers(weights):
    new_keys = []
    for key, value in unpickled_model_wts.items():
        new_keys.append(key.replace('module.', ''))
    for i in new_keys:
        unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)#    
    return unpickled_model_wts

def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                print('conv')
                #torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
            	print('inv')
#                for idx in range(len(m.conv)):
#                    if type(m.conv[idx]) == nn.Conv2d:
#                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)
################################################
model_fp32 = models.quantization.resnet18(weights=None)

pretrained_model_wts = pickle.load(open(os.path.join(root, 'CocoaNet18_DN.pkl'), "rb"))
unpickled_model_wts = copy.deepcopy(pretrained_model_wts['model'])
unpickled_model_wts = Remove_module_from_layers(unpickled_model_wts)

in_feat = model_fp32.fc.in_features
out_feat = unpickled_model_wts['fc.weight'].size()[0]
model_fp32.fc = nn.Linear(in_feat, out_feat)
#Load custom weights
model_fp32.load_state_dict(unpickled_model_wts)
#Delete final linear layer and replace to match n classes in the dataset
model_fp32.fc = torch.nn.Linear(in_feat, num_classes)

###############################

model_fp32.eval()
model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

#for m in model_fp32.modules():
#    print(m)

model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv1', 'bn1', 'relu']])


model_fp32_prepared = torch.quantization.prepare(model_fp32)#
model_fp32_prepared(input_fp32)
model_int8 = torch.quantization.convert(model_fp32_prepared)
res = model_int8(input_fp32)#
print(res)


save_path = os.path.join(root, 'CocoaNet18_DN_PTSQuantised')
torchscript_model = torch.jit.script(model_int8)
torch.jit.save(torchscript_model, save_path + ".pth")
optimized_torchscript_model = optimize_for_mobile(torchscript_model)
optimized_torchscript_model.save(save_path + "_mobile.pth")
optimized_torchscript_model._save_for_lite_interpreter(save_path + "_mobile.ptl")