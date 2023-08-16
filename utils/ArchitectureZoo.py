#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
from torchvision.ops import SqueezeExcitation



class CNBlock_pico(nn.Module):
    def __init__(self, dim, kernel_3, kernel_4, layer_scale=0.3, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_3, padding='same', groups=dim, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_4, padding='same'),
            nn.Dropout(dropout),  
        )
        nn.init.kaiming_normal_(self.block[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.block[2].weight, mode='fan_out', nonlinearity='relu')

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result += input
        return result



# class DisNet_pico(nn.Module):
#     def __init__(self, out_channels, config_dict):
#         super(DisNet_pico, self).__init__()
#         self.config_dict = config_dict

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=config_dict['dim_1'], kernel_size=config_dict['kernel_1'], padding='same') 
#         nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
#         self.cnblock1 = CNBlock_pico(dim=config_dict['dim_1'], kernel_3=config_dict['kernel_3'], kernel_4=config_dict['kernel_4'], dropout=config_dict['drop_out'])
#         self.Avgpool = nn.AvgPool2d(2, 2)
#         self.Maxpool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(in_channels=config_dict['dim_1'], out_channels=config_dict['dim_2'], kernel_size=config_dict['kernel_2'], padding='same')
#         nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
#         self.cnblock2 = CNBlock_pico(dim=config_dict['dim_2'], kernel_3=config_dict['kernel_5'], kernel_4=config_dict['kernel_6'], dropout=config_dict['drop_out'])

#         self.fc1 = None
#         #self.fc1 = nn.Linear(150000, self.config_dict['nodes_1'])
        
#         self.fc2 = nn.Linear(config_dict['nodes_1'], config_dict['nodes_2'])
#         nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
#         self.fc3 = nn.Linear(config_dict['nodes_2'], out_channels)
#         nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):
#         x = F.relu(self.conv1(x))   
#         x = self.cnblock1(x)        
#         x = self.Maxpool(x)            
#         x = F.relu(self.conv2(x))   
#         x = self.cnblock2(x)        
#         x = self.Avgpool(x)     
#         x = torch.flatten(x, 1)   
          
#         if self.fc1 is None:
#             self.fc1 = nn.Linear(x.shape[1], self.config_dict['nodes_1']).to(x.device)
#             nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')        
#         x = F.gelu(self.fc1(x))     
#         x = F.gelu(self.fc2(x))     
#         x = self.fc3(x)             
#         return x




# class DisNet_pico_duo(nn.Module):
#     config_dict = {
#         "dim_1": 34,
#         "dim_2": 25,
#         "drop_out": 0.15764421413342755,
#         "input_size": 388,
#         "kernel_1": 3,
#         "kernel_2": 4,
#         "kernel_3": 3,
#         "kernel_4": 3,
#         "kernel_5": 4,
#         "kernel_6": 7,
#         "nodes_1": 80,
#         "nodes_2": 65,
#     }


#     def __init__(self, out_channels):
#         super(DisNet_pico_duo, self).__init__()
#         # self.config_dict = config_dict

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.config_dict['dim_1'], kernel_size=self.config_dict['kernel_1'], padding='same') 
#         nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        

#         self.cnblock1 = CNBlock_pico(dim=self.config_dict['dim_1'], kernel_3=self.config_dict['kernel_3'], kernel_4=self.config_dict['kernel_4'], dropout=self.config_dict['drop_out'])
#         self.se1 = SqueezeExcitation(input_channels=self.config_dict['dim_1'], squeeze_channels=1)
        
#         self.conv2 = nn.Conv2d(in_channels=self.config_dict['dim_1'], out_channels=self.config_dict['dim_2'], kernel_size=self.config_dict['kernel_2'], padding='same')
#         nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        
#         self.cnblock2 = CNBlock_pico(dim=self.config_dict['dim_2'], kernel_3=self.config_dict['kernel_5'], kernel_4=self.config_dict['kernel_6'], dropout=self.config_dict['drop_out'])

#         # self.fc1 = None
#         self.fc1 = nn.Linear(235225, self.config_dict['nodes_1'])
        
#         self.fc2 = nn.Linear(self.config_dict['nodes_1'], self.config_dict['nodes_2'])
#         nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
#         self.fc2_norm = nn.LayerNorm(self.config_dict['nodes_2']) # Layer normalization

#         self.fc3_1 = nn.Linear(self.config_dict['nodes_2'], out_channels)
#         nn.init.kaiming_normal_(self.fc3_1.weight, mode='fan_out', nonlinearity='relu')
#         # self.fc3_1_norm = nn.LayerNorm(out_channels) # Layer normalization

#         self.fc3_2 = nn.Linear(self.config_dict['nodes_2'], 2)
#         nn.init.kaiming_normal_(self.fc3_2.weight, mode='fan_out', nonlinearity='relu')
#         # self.fc3_2_norm = nn.LayerNorm(2) # Layer normalization
        
#         self.Maxpool = nn.MaxPool2d(2, 2) 
#         self.Avgpool = nn.AvgPool2d(2, 2)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.cnblock1(x)
#         x = self.se1(x)         
#         x = self.Maxpool(x)            
#         x = self.conv2(x)   
#         x = F.relu(x)
#         x = self.cnblock2(x)   
#         x = self.Avgpool(x)     
#         x = torch.flatten(x, 1)     
          
#         if self.fc1 is None:
#             self.fc1 = nn.Linear(x.shape[1], self.config_dict['nodes_1']).to(x.device)
#             nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
    
#         x = F.gelu(self.fc1(x))     
#         x = F.gelu(self.fc2_norm(self.fc2(x)))     
#         y = self.fc3_1(x) 
#         z = self.fc3_2(x)        
#         return y, z

class DisNet_pico(nn.Module):
    # config_dict = {
    # "dim_1": 21,
    # "dim_2": 41,
    # "drop_out": 0.1280985437968713,
    # "input_size": 455,
    # "kernel_1": 2,
    # "kernel_2": 1,
    # "kernel_3": 8,
    # "kernel_4": 4,
    # "kernel_5": 4,
    # "kernel_6": 4,
    # "nodes_1": 84,
    # "nodes_2": 79,
    # }


    def __init__(self, out_channels, config):
        super(DisNet_pico, self).__init__()
        self.config_dict = config
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.config_dict['dim_1'], kernel_size=self.config_dict['kernel_1'], padding='same') 
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        

        self.cnblock1 = CNBlock_pico(dim=self.config_dict['dim_1'], kernel_3=self.config_dict['kernel_3'], kernel_4=self.config_dict['kernel_4'], dropout=self.config_dict['drop_out'])
        self.se1 = SqueezeExcitation(input_channels=self.config_dict['dim_1'], squeeze_channels=1)
        
        self.conv2 = nn.Conv2d(in_channels=self.config_dict['dim_1'], out_channels=self.config_dict['dim_2'], kernel_size=self.config_dict['kernel_2'], padding='same')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        
        self.cnblock2 = CNBlock_pico(dim=self.config_dict['dim_2'], kernel_3=self.config_dict['kernel_5'], kernel_4=self.config_dict['kernel_6'], dropout=self.config_dict['drop_out'])

        self.fc1 = None
        # self.fc1 = nn.Linear(523529, self.config_dict['nodes_1'])
        
        self.fc2 = nn.Linear(self.config_dict['nodes_1'], self.config_dict['nodes_2'])
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        self.fc2_norm = nn.LayerNorm(self.config_dict['nodes_2']) 
        
        self.fc3 = nn.Linear(self.config_dict['nodes_2'], out_channels)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        
        self.Maxpool = nn.MaxPool2d(2, 2) 
        self.Avgpool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.cnblock1(x)
        x = self.se1(x)         
        x = self.Maxpool(x)            
        x = self.conv2(x)   
        x = F.relu(x)
        x = self.cnblock2(x)   
        x = self.Avgpool(x)     
        x = torch.flatten(x, 1)     
          
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], self.config_dict['nodes_1']).to(x.device)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
    
        x = F.gelu(self.fc1(x))     
        x = F.gelu(self.fc2_norm(self.fc2(x)))     
        x = self.fc3(x) 
        return x
    

class DisNet_picoIR(nn.Module):
    config_dict = {
    "dim_1": 21,
    "dim_2": 41,
    "drop_out": 0.1280985437968713,
    "input_size": 455,
    "kernel_1": 2,
    "kernel_2": 1,
    "kernel_3": 8,
    "kernel_4": 4,
    "kernel_5": 4,
    "kernel_6": 4,
    "nodes_1": 84,
    "nodes_2": 79,
    }


    def __init__(self, out_channels):
        super(DisNet_picoIR, self).__init__()
        # self.config_dict = config
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.config_dict['dim_1'], kernel_size=self.config_dict['kernel_1'], padding='same') 
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        

        self.cnblock1 = CNBlock_pico(dim=self.config_dict['dim_1'], kernel_3=self.config_dict['kernel_3'], kernel_4=self.config_dict['kernel_4'], dropout=self.config_dict['drop_out'])
        self.se1 = SqueezeExcitation(input_channels=self.config_dict['dim_1'], squeeze_channels=1)
        
        self.conv2 = nn.Conv2d(in_channels=self.config_dict['dim_1'], out_channels=self.config_dict['dim_2'], kernel_size=self.config_dict['kernel_2'], padding='same')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        
        self.cnblock2 = CNBlock_pico(dim=self.config_dict['dim_2'], kernel_3=self.config_dict['kernel_5'], kernel_4=self.config_dict['kernel_6'], dropout=self.config_dict['drop_out'])

        # self.fc1 = None
        self.fc1 = nn.Linear(523529, self.config_dict['nodes_1'])
        
        self.fc2 = nn.Linear(self.config_dict['nodes_1'], self.config_dict['nodes_2'])
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        self.fc2_norm = nn.LayerNorm(self.config_dict['nodes_2']) 
        
        self.fc3 = nn.Linear(self.config_dict['nodes_2'], out_channels)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
        
        self.Maxpool = nn.MaxPool2d(2, 2) 
        self.Avgpool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.cnblock1(x)
        x = self.se1(x)         
        x = self.Maxpool(x)            
        x = self.conv2(x)   
        x = F.relu(x)
        x = self.cnblock2(x)   
        x = self.Avgpool(x)     
        x = torch.flatten(x, 1)     
          
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], self.config_dict['nodes_1']).to(x.device)
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
    
        x = F.gelu(self.fc1(x))     
        x = F.gelu(self.fc2_norm(self.fc2(x)))     
        x = self.fc3(x) 
        return x


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
