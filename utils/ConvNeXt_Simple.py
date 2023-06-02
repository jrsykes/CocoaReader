#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.convnext import LayerNorm2d
from torchvision.ops.stochastic_depth import StochasticDepth
import sys
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
from ColorGradingLayer import CrossTalkColorGrading


class CNBlock(nn.Module):
    def __init__(self, dim, kernel_3, kernel_4, layer_scale=0.3, stochastic_depth_prob=0.001):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_3, padding='same', groups=dim, bias=True),
            LayerNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_4, padding='same'),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode="row")


    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result

class ConvNeXt_simple(nn.Module):
    def __init__(self, config_dict = None):
        super(ConvNeXt_simple, self).__init__()

        self.color_grading = CrossTalkColorGrading(matrix='Best')
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=config_dict['dim_1'], kernel_size=config_dict['kernel_1'], padding='same') 
        self.cnblock1 = CNBlock(dim=config_dict['dim_1'], kernel_3=config_dict['kernel_3'], kernel_4=config_dict['kernel_4'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=config_dict['dim_1'], out_channels=config_dict['dim_2'], kernel_size=config_dict['kernel_2'], padding='same')
        self.cnblock2 = CNBlock(dim=config_dict['dim_2'], kernel_3=config_dict['kernel_5'], kernel_4=config_dict['kernel_6'], layer_scale=config_dict['layer_scale'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])

        in_ = config_dict['dim_2'] * 100 * 100

        self.fc1 = nn.Linear(in_, config_dict['nodes_1'])
        self.fc2 = nn.Linear(config_dict['nodes_1'], config_dict['nodes_2'])
        self.fc3 = nn.Linear(config_dict['nodes_2'], config_dict['num_classes'])

    def forward(self, x):
        x = self.color_grading(x) # apply cross channel channel color grading
        x = F.relu(self.conv1(x))   
        x = self.cnblock1(x)        
        x = self.pool(x)            
        x = F.relu(self.conv2(x))   
        x = self.cnblock2(x)        
        x = self.pool(x)            
        x = torch.flatten(x, 1)     
        x = F.gelu(self.fc1(x))     
        x = F.gelu(self.fc2(x))     
        x = self.fc3(x)             
        return x


class DisNet_nano(nn.Module):
    def __init__(self, config_dict = None):
        super(DisNet_nano, self).__init__()

        self.color_grading = CrossTalkColorGrading(matrix='Best')
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=config_dict['dim_1'], kernel_size=config_dict['kernel_1'], padding='same') 
        self.cnblock1 = CNBlock(dim=config_dict['dim_1'], kernel_3=config_dict['kernel_3'], kernel_4=config_dict['kernel_4'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=config_dict['dim_1'], out_channels=config_dict['dim_2'], kernel_size=config_dict['kernel_2'], padding='same')
        self.cnblock2 = CNBlock(dim=config_dict['dim_2'], kernel_3=config_dict['kernel_5'], kernel_4=config_dict['kernel_6'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])

        self.conv3 = nn.Conv2d(in_channels=config_dict['dim_2'], out_channels=config_dict['dim_3'], kernel_size=config_dict['kernel_2'], padding='same')
        self.cnblock3 = CNBlock(dim=config_dict['dim_3'], kernel_3=config_dict['kernel_7'], kernel_4=config_dict['kernel_8'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])

        in_ = config_dict['dim_3'] * 100 * 100

        self.fc1 = nn.Linear(in_, config_dict['nodes_1'])
        self.fc2 = nn.Linear(config_dict['nodes_1'], config_dict['nodes_2'])
        self.fc3 = nn.Linear(config_dict['nodes_2'], config_dict['num_classes'])

    def forward(self, x):
        x = self.color_grading(x) # apply cross channel channel color grading
        x = F.relu(self.conv1(x))   
        x = self.cnblock1(x)        
        x = self.pool(x)            
        x = F.relu(self.conv2(x))   
        x = self.cnblock2(x)        
        x = self.pool(x) 

        x = F.relu(self.conv3(x))   
        x = self.cnblock3(x)             
        
        x = torch.flatten(x, 1)     
        x = F.gelu(self.fc1(x))     
        x = F.gelu(self.fc2(x))     
        x = self.fc3(x)             
        return x

# config_dict = {
#     'dim_1': 36,
#     'dim_2': 64,
#     'dim_3': 64,
#     'kernel_1': 3,
#     'kernel_2': 1,
#     'kernel_3': 4,
#     'kernel_4': 2,
#     'kernel_5': 3,
#     'kernel_6': 2,
#     'kernel_7': 3,
#     'kernel_8': 2,
#     'layer_scale': 0.3,
#     'stochastic_depth_prob': 0.001,
#     'nodes_1': 91,
#     'nodes_2': 132,
#     'num_classes': 2
# }
# print()
# print()
# #print numbe of parameters
# #print(f'DisNet parameters: {sum(p.numel() for p in model.parameters())}')
# #print number of floating point opperations
# from pthflops import count_ops

# # Create a random input tensor
# input = torch.randn(1, 3, 400, 400)


# model = ConvNeXt_simple(config_dict)

# # Calculate FLOPs
# flopts = count_ops(model, input)
# print(f'DisNet FLOPs: {flopts}')


# model = DisNet_nano(config_dict)

# # Calculate FLOPs
# flopts = count_ops(model, input)
# print(f'DisNet FLOPs: {flopts}')

# from torchvision import models

# #load resnet18 model
# model = models.resnet18(pretrained=None)
# #print(f'resNet18 parameters: {sum(p.numel() for p in model.parameters())}')
# flopts = count_ops(model, input)
# print()
# print(f'resnet FLOPs: {flopts}')

# #load convnext_tiny model
# model = models.convnext_tiny(pretrained=None)
# #print(f'ConvNext parameters: {sum(p.numel() for p in model.parameters())}')
# flopts = count_ops(model, input)
# print()
# print(f'convext FLOPs: {flopts}')
# # ############

# #%%


