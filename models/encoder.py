import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size=256, scale=1.0):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.mapping_size = mapping_size
        self.scale = scale
        
        self.register_buffer(
            'B',
            torch.randn((num_input_channels, mapping_size)) * scale
        )
    
    def forward(self, x):
        # x: batch_size x num_input_channels
        x_proj = (2 * torch.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)