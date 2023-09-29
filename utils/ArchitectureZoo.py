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
  

class DisNet(nn.Module):

    def __init__(self, out_channels, config=None):
        super(DisNet, self).__init__()
        self.config_dict = config
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.config_dict['dim_1'], kernel_size=self.config_dict['kernel_1'], padding='same') 
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        

        self.cnblock1 = CNBlock_pico(dim=self.config_dict['dim_1'], kernel_3=self.config_dict['kernel_3'], kernel_4=self.config_dict['kernel_4'], dropout=self.config_dict['drop_out'])
        self.se1 = SqueezeExcitation(input_channels=self.config_dict['dim_1'], squeeze_channels=1)
        
        self.conv2 = nn.Conv2d(in_channels=self.config_dict['dim_1'], out_channels=self.config_dict['dim_2'], kernel_size=self.config_dict['kernel_2'], padding='same')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        
        self.cnblock2 = CNBlock_pico(dim=self.config_dict['dim_2'], kernel_3=self.config_dict['kernel_5'], kernel_4=self.config_dict['kernel_6'], dropout=self.config_dict['drop_out'])

        # self.fc1 = None
        self.fc1 = nn.Linear(39690, self.config_dict['nodes_1'])
        
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


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, max_groups=32, kernel_size=3):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // self.expansion
        groups = mid_channels if mid_channels % max_groups == 0 else 1  # Ensure num_groups is a divisor of num_channels
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False, groups=groups),
            nn.GroupNorm(num_groups=groups, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=groups, num_channels=out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out_residual = self.residual(x)
        out_shortcut = self.shortcut(x)
        
        # Check if padding is needed
        if out_residual.size(2) != out_shortcut.size(2) or out_residual.size(3) != out_shortcut.size(3):
            padding = [0, out_residual.size(3) - out_shortcut.size(3), 0, out_residual.size(2) - out_shortcut.size(2)]
            out_shortcut = nn.functional.pad(out_shortcut, padding)
        
        out = out_residual + out_shortcut
        out = self.relu(out)
        return out


class DisNetV1_2(nn.Module):
    def __init__(self, config=None):
        super(DisNetV1_2, self).__init__()
        self.config_dict = config
        
        out_channels = self.config_dict['out_channels']
        groups = self.config_dict['dim_1'] if self.config_dict['dim_1'] % 32 == 0 else 1  # Ensure num_groups is a divisor of num_channels
        
        self.conv1 = nn.Conv2d(3, self.config_dict['dim_1'], kernel_size=self.config_dict['kernel_1'], stride=2, padding=self.config_dict['kernel_1']//2, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=self.config_dict['dim_1'])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Pass num_blocks parameter when creating layers
        self.layer1 = self._make_layer(self.config_dict['dim_1'], self.config_dict['dim_2'], blocks=self.config_dict['num_blocks_1'], stride=1, kernel_size=self.config_dict['kernel_2'])
        self.layer2 = self._make_layer(self.config_dict['dim_2'], self.config_dict['dim_3'], blocks=self.config_dict['num_blocks_2'], stride=2, kernel_size=self.config_dict['kernel_3'])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.config_dict['dim_3'], out_channels)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride, kernel_size):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride=stride, kernel_size=kernel_size))
        for _ in range(1, blocks):  # Use blocks parameter to determine the number of Bottleneck blocks
            layers.append(Bottleneck(out_channels, out_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)
         
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x



# class AttentionNet(nn.Module):
#     def __init__(self, num_classes, num_tokens):
#         super(AttentionNet, self).__init__()
#         num_heads = num_classes
#         embed_dim = num_tokens**2*num_classes
#         head_dim = embed_dim//num_heads
#         # define linear transformations for queries, keys, and values
#         self.query_transform = nn.Linear(embed_dim, num_heads * head_dim)
#         self.key_transform = nn.Linear(embed_dim, num_heads * head_dim)
#         self.value_transform = nn.Linear(embed_dim, num_heads * head_dim)
#         #apply muti-AttentionNet head
#         self.Attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
#         #apply layer normalisation
#         self.layernorm = nn.LayerNorm(embed_dim)
#         #apply linear layer
#         self.fc1 = nn.Linear(embed_dim, embed_dim)
#         self.fc2 = nn.Linear(embed_dim, num_classes)


#     def forward(self, x):
#         #apply AttentionNet
#         queries = self.query_transform(x)
#         keys = self.key_transform(x)
#         values = self.value_transform(x)
#         x, _ = self.Attention(queries, keys, values)
#         x = self.fc1(x)
#         #apply layer normalisation
#         x = self.layernorm(x)
#         #apply linear layer
#         x = self.fc2(x)
#         #apply relu activation
#         x = F.softmax(x, dim=1)
#         return x

# Define the meta-model
class MetaModel(nn.Module):
    def __init__(self, config):
        super(MetaModel, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config['trans_nodes']*2, self.config['nodes_3'])
        self.bn1 = nn.LayerNorm(self.config['nodes_3'])
        self.fc2 = nn.Linear(self.config['nodes_3'], self.config['nodes_4'])
        self.bn2 = nn.LayerNorm(self.config['nodes_4'])
        self.fc3 = nn.Linear(self.config['nodes_4'], self.config['num_classes'])
        self.dropout = nn.Dropout(self.config['drop_out2'])
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# Define the unified model
class UnifiedModel(nn.Module):
    def __init__(self, CNN1, CNN2, MetaModel):
        super(UnifiedModel, self).__init__()
        self.cnn1 = CNN1
        self.cnn2 = CNN2
        self.meta = MetaModel

    def forward(self, x):
        out1 = self.cnn1(x)
        out2 = self.cnn2(x)
        merged = torch.cat((out1, out2), dim=1)
        return self.meta(merged)