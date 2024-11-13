#%%
import torch
import torch.nn as nn
import sys

sys.path.append('~/scripts/CocoaReader/utils')

class Bottleneck_ablation(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, max_groups=32, kernel_size=3):
        super(Bottleneck_ablation, self).__init__()
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


class PhytNetV0_ablation(nn.Module):
    def __init__(self, config):
        super(PhytNetV0_ablation, self).__init__()
        self.config_dict = config
        
        out_channels = self.config_dict['out_channels']
        groups = self.config_dict['dim_1'] if self.config_dict['dim_1'] % 32 == 0 else 1  # Ensure num_groups is a divisor of num_channels
        
        self.conv1 = nn.Conv2d(3, self.config_dict['dim_1'], kernel_size=self.config_dict['kernel_1'], stride=2, padding=self.config_dict['kernel_1']//2, bias=False)
        
        # self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=self.config_dict['dim_1'])
        self.gn1 = nn.LayerNorm((49, 143, 143))
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Pass num_blocks parameter when creating layers
        self.layer1 = self._make_layer(self.config_dict['dim_1'], self.config_dict['dim_2'], blocks=self.config_dict['num_blocks_1'], stride=1, kernel_size=self.config_dict['kernel_2'])
        self.layer2 = self._make_layer(self.config_dict['dim_2'], self.config_dict['dim_3'], blocks=self.config_dict['num_blocks_2'], stride=2, kernel_size=self.config_dict['kernel_3'])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.config_dict['dim_3'], out_channels)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride, kernel_size):
        layers = []
        layers.append(Bottleneck_ablation(in_channels, out_channels, stride=stride, kernel_size=kernel_size))
        for _ in range(1, blocks):  # Use blocks parameter to determine the number of Bottleneck blocks
            layers.append(Bottleneck_ablation(out_channels, out_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)
         
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)       

        w = self.global_avg_pool(x)
        w = torch.flatten(w, 1)      
        y = self.fc(w)
        
        return x, w, y

