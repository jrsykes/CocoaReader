#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision.transforms as transforms

sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')


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
    def __init__(self, config):
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
        
        w = self.global_avg_pool(x)
        w = torch.flatten(w, 1)
        
        # x = self.fc(x)
        
        return x, w



class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding2D, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        
        # Handling the case when d_model is odd
        if d_model % 2 == 0:
            self.encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            self.encoding[:, 1::2] = torch.cos(position * div_term)[:,:-1]
        
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)
        y = self.encoding[:x.size(0), :].to(x.device)
        return x + y

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, memory):
        attn_output, _ = self.self_attn(x, x, x)  # Self attention
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        fc_output = self.fc(x)
        x = x + self.dropout(fc_output)
        x = self.norm2(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, img_size):
        super(TransformerDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.img_size = img_size
        self.positional_encoding = PositionalEncoding2D(feature_dim)
        self.decoder_blocks = nn.ModuleList([TransformerDecoderBlock(feature_dim, num_heads) for _ in range(num_layers)])
        
        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(feature_dim, feature_dim, kernel_size=3, stride=3, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.final_conv = nn.Conv2d(feature_dim, 3, kernel_size=1)  # Assuming the output has 3 channels

    def forward(self, x):
        # x is the encoded feature map: [batch_size, feature_dim, height, width]
        batch_size, _, height, width = x.size()
        x = self.positional_encoding(x)
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x)  # Pass through each transformer block
        
        # Upsampling steps
        x = self.upsample1(x.permute(1, 2, 0).view(batch_size, self.feature_dim, height, width))
        x = self.upsample2(x)
        x = self.upsample3(x)
        
        x = self.final_conv(x)
        return x



class DisNet_SRAutoencoder(nn.Module):
    def __init__(self, config):
        super(DisNet_SRAutoencoder, self).__init__()
        
        # Initialize the encoder
        self.encoder = DisNetV1_2(config)
        
        # Extract necessary parameters from config or define them manually
        feature_dim = config['dim_3']
        num_heads = config['num_heads'] 
        num_layers = config['num_decoder_layers']
        img_size = config['input_size'] 
        
        # Initialize the transformer decoder
        self.decoder = TransformerDecoder(feature_dim, num_heads, num_layers, img_size)


    def forward(self, x):
        
        encoded, encode_pooled  = self.encoder.forward(x)  # Get feature map before the fc layer
        decoded = self.decoder(encoded)

        return encode_pooled, decoded 



##################################################

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
                
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.bottleneck = self.conv_block(128, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        dec2 = self.decoder2(torch.cat([enc2, self.upconv2(bottleneck)], 1))
        dec1 = self.decoder1(torch.cat([enc1, self.upconv1(dec2)], 1))
        
        return self.final_conv(dec1)


from diffusers import UNet2DModel
from torch.nn import functional as F




class PhytNet_SRAutoencoder(nn.Module):
    def __init__(self, config):
        super(PhytNet_SRAutoencoder, self).__init__()
        
        # Initialize the encoder (assuming DisNetV1_2 is defined elsewhere)
        self.encoder = DisNetV1_2(config)
        
        # Extract necessary parameters from config or define them manually
        feature_dim = config['dim_3']
        num_heads = config['num_heads'] 
        num_layers = config['num_decoder_layers']
        img_size = config['input_size'] 
        
        # Initialize the transformer decoder (assuming TransformerDecoder is defined elsewhere)
        self.decoder = TransformerDecoder(feature_dim, num_heads, num_layers, img_size)
        
        # Diffusion parameters
        self.noise_std = config.get('noise_std', 0.2)  # Default noise standard deviation
        self.num_diffusion_steps = config.get('num_diffusion_steps', 10)  # Default number of diffusion steps
        
        # Add U-Net for refinement
        # self.refinement_net = UNet(feature_dim, feature_dim)
        self.refinement_net = UNet2DModel(
                                    sample_size=(48, 48),
                                    in_channels=84,
                                    out_channels=84).float()

    def latent_diffusion(self, encoded):
        # Add noise to the encoded representations
        noisy_encoded = encoded + torch.randn_like(encoded) * self.noise_std
        noisy_encoded = F.pad(noisy_encoded, (1, 2, 1, 2))

        # Iteratively refine the noisy representations using the U-Net
        for i in range(self.num_diffusion_steps):
            noisy_encoded = self.refinement_net(sample=noisy_encoded, timestep=i)
            noisy_encoded = noisy_encoded.sample
        
        noisy_encoded = noisy_encoded[:, :, :45, :45]

        return noisy_encoded

    def forward(self, x):
        encoded, encode_pooled  = self.encoder.forward(x)  # Get feature map before the fc layer
            
        # Apply latent diffusion to the encoded representations
        diffused_encoded = self.latent_diffusion(encoded)
        
        decoded = self.decoder(diffused_encoded)
        return encode_pooled, decoded 
    

