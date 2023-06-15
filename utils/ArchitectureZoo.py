#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.convnext import LayerNorm2d
from torchvision.ops.stochastic_depth import StochasticDepth
import sys
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
from ColorGradingLayer import CrossTalkColorGrading


class CNBlock_nano(nn.Module):
    def __init__(self, dim, kernel_3, kernel_4, layer_scale=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_3, padding='same', groups=dim, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_4, padding='same'),
            nn.Dropout(0.5)  
            )
        # Apply Kaiming/He Initialization to Conv2d layers
        nn.init.kaiming_normal_(self.block[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.block[2].weight, mode='fan_out', nonlinearity='relu')

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result += input
        return result

class CNBlock_pico(nn.Module):
    def __init__(self, dim, kernel_3, kernel_4, layer_scale=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_3, padding='same', groups=dim, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_4, padding='same'),
            nn.Dropout(0.5),  
        )
        nn.init.kaiming_normal_(self.block[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.block[2].weight, mode='fan_out', nonlinearity='relu')

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result += input
        return result

class DisNet_nano(nn.Module):
    def __init__(self, out_channels):
        super(DisNet_nano, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=5, padding='same')
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.cnblock1 = CNBlock_nano(dim=15, kernel_3=7, kernel_4=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=22, kernel_size=1, padding='same')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.cnblock2 = CNBlock_nano(dim=22, kernel_3=2, kernel_4=1)
 
        self.fc1 = None
        self.fc2 = nn.Linear(102, 103)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        self.fc3 = nn.Linear(103, out_channels)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
    

    def forward(self, x):
        x = F.relu(self.conv1(x))   
        x = self.cnblock1(x)        
        x = self.pool(x)            
        x = F.relu(self.conv2(x))   
        x = self.cnblock2(x)        
        x = self.pool(x)     
        x = torch.flatten(x, 1)         

        # Dynamically calculate the input size for fc1
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 102).to(x.device)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')

        x = F.gelu(self.fc1(x))  
        x = F.gelu(self.fc2(x))  
        x = self.fc3(x)             
        return x




class DisNet_pico(nn.Module):
    def __init__(self, out_channels):
        super(DisNet_pico, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding='same')
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.cnblock1 = CNBlock_pico(dim=10, kernel_3=7, kernel_4=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding='same')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.cnblock2 = CNBlock_pico(dim=10, kernel_3=5, kernel_4=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=1, padding='same')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.cnblock3 = CNBlock_pico(dim=15, kernel_3=2, kernel_4=1)
        
        self.fc1 = None
        self.fc2 = nn.Linear(70, 35)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        self.ln = nn.LayerNorm(35)
        self.fc3 = nn.Linear(35, out_channels)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.conv1(x))   
        x = self.cnblock1(x)        
        x = F.relu(self.conv2(x))   
        x = self.cnblock2(x)        
        x = self.pool(x)            
        x = F.relu(self.conv3(x))   
        x = self.cnblock3(x)        
        x = self.pool(x)     
        x = torch.flatten(x, 1)         

        # Dynamically calculate the input size for fc1
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 70).to(x.device)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')

        x = F.gelu(self.fc1(x))  
        x = F.gelu(self.fc2(x))
        x = self.ln(x)     
        x = self.fc3(x)             
        return x








# class DisNet_pico(nn.Module):
#     def __init__(self, config_dict = None, IR = False):
#         super(DisNet_pico, self).__init__()
#         self.config_dict = config_dict
#         self.IR = IR
#         if self.IR == True:
#             self.color_grading = CrossTalkColorGrading(matrix='Best')
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=config_dict['dim_1'], kernel_size=config_dict['kernel_1'], padding='same') 
#         nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
#         self.cnblock1 = CNBlock(dim=config_dict['dim_1'], kernel_3=config_dict['kernel_3'], kernel_4=config_dict['kernel_4'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])
#         self.pool = nn.AvgPool2d(2, 2)
#         self.conv2 = nn.Conv2d(in_channels=config_dict['dim_1'], out_channels=config_dict['dim_2'], kernel_size=config_dict['kernel_2'], padding='same')
#         nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
#         self.cnblock2 = CNBlock(dim=config_dict['dim_2'], kernel_3=config_dict['kernel_5'], kernel_4=config_dict['kernel_6'], layer_scale=config_dict['layer_scale'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])

#         self.fc1 = None
#         self.fc2 = nn.Linear(config_dict['nodes_1'], config_dict['nodes_2'])
#         nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
#         self.fc3 = nn.Linear(config_dict['nodes_2'], config_dict['num_classes'])
#         nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):
#         if self.IR == True:
#             x = self.color_grading(x) # apply cross channel channel color grading
#         x = F.relu(self.conv1(x))   
#         x = self.cnblock1(x)        
#         x = self.pool(x)            
#         x = F.relu(self.conv2(x))   
#         x = self.cnblock2(x)        
#         x = self.pool(x)     

#         x = torch.flatten(x, 1)     
#         if self.fc1 is None:
#             self.fc1 = nn.Linear(x.shape[1], self.config_dict['nodes_1']).to(x.device)
#             nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')        
#         x = F.gelu(self.fc1(x))     
#         x = F.gelu(self.fc2(x))     
#         x = self.fc3(x)             
#         return x







#Needs work 
# class DisNet_nano(nn.Module):
#     def __init__(self, config_dict, seed=42, IR = False):
#         super(DisNet_nano, self).__init__()
#         self.IR = IR

#         # Set the seed for consistency
#         torch.manual_seed(seed)

#         self.color_grading = CrossTalkColorGrading(matrix='Best')
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=config_dict['dim_1'], kernel_size=config_dict['kernel_1'], padding='same') 
#         nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
#         self.cnblock1 = CNBlock(dim=config_dict['dim_1'], kernel_3=config_dict['kernel_3'], kernel_4=config_dict['kernel_4'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])
#         self.pool = nn.AvgPool2d(2, 2)
#         self.conv2 = nn.Conv2d(in_channels=config_dict['dim_1'], out_channels=config_dict['dim_2'], kernel_size=config_dict['kernel_2'], padding='same')
#         nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
#         self.cnblock2 = CNBlock(dim=config_dict['dim_2'], kernel_3=config_dict['kernel_5'], kernel_4=config_dict['kernel_6'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])
#         self.conv3 = nn.Conv2d(in_channels=config_dict['dim_2'], out_channels=config_dict['dim_3'], kernel_size=config_dict['kernel_2'], padding='same')
#         nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
#         self.cnblock3 = CNBlock(dim=config_dict['dim_3'], kernel_3=config_dict['kernel_7'], kernel_4=config_dict['kernel_8'], stochastic_depth_prob=config_dict['stochastic_depth_prob'])

#         in_ = config_dict['dim_3'] * 100 * 100

#         self.fc1 = nn.Linear(in_, config_dict['nodes_1'])
#         nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
#         self.fc2 = nn.Linear(config_dict['nodes_1'], config_dict['nodes_2'])
#         nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
#         self.fc3 = nn.Linear(config_dict['nodes_2'], config_dict['num_classes'])
# #         nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):
#         if self.IR == True:
#             x = self.color_grading(x) # apply cross channel channel color grading
#         x = F.relu(self.conv1(x))   
#         x = self.cnblock1(x)        
#         x = self.pool(x)            
#         x = F.relu(self.conv2(x))   
#         x = self.cnblock2(x)        
#         x = self.pool(x) 

#         x = F.relu(self.conv3(x))   
#         x = self.cnblock3(x)             
        
#         x = torch.flatten(x, 1)     
#         x = F.gelu(self.fc1(x))     
#         x = F.gelu(self.fc2(x))     
#         x = self.fc3(x)             
#         return





class AttentionNet(nn.Module):
    def __init__(self, num_classes, num_tokens):
        super(AttentionNet, self).__init__()
        num_heads = num_classes
        embed_dim = num_tokens**2*num_classes
        head_dim = embed_dim//num_heads
        # define linear transformations for queries, keys, and values
        self.query_transform = nn.Linear(embed_dim, num_heads * head_dim)
        self.key_transform = nn.Linear(embed_dim, num_heads * head_dim)
        self.value_transform = nn.Linear(embed_dim, num_heads * head_dim)
        #apply muti-AttentionNet head
        self.Attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        #apply layer normalisation
        self.layernorm = nn.LayerNorm(embed_dim)
        #apply linear layer
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        #apply AttentionNet
        queries = self.query_transform(x)
        keys = self.key_transform(x)
        values = self.value_transform(x)
        x, _ = self.Attention(queries, keys, values)
        x = self.fc1(x)
        #apply layer normalisation
        x = self.layernorm(x)
        #apply linear layer
        x = self.fc2(x)
        #apply relu activation
        x = F.softmax(x, dim=1)
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

# input_size = 300
# model = DisNet_pico(config_dict)

# print()
# print()
# #print numbe of parameters
# #print(f'DisNet parameters: {sum(p.numel() for p in model.parameters())}')
# #print number of floating point opperations
# from pthflops import count_ops


# # Create a random input tensor
# input = torch.randn(1, 3, input_size, input_size)

# out = model(input)

# print(out)


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


