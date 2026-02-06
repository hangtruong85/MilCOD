"""
CBAM: Convolutional Block Attention Module
Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Focuses on "what" is meaningful
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        
        # Shared MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP layers
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: input features (B, C, H, W)
        Returns:
            channel attention map (B, C, 1, 1)
        """
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling branch
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focuses on "where" is meaningful
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Spatial attention convolution
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: input features (B, C, H, W)
        Returns:
            spatial attention map (B, 1, H, W)
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        out = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        out = self.sigmoid(self.conv(out))
        
        return out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Sequential application of channel and spatial attention
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        """
        Args:
            x: input features (B, C, H, W)
        Returns:
            attention-refined features (B, C, H, W)
        """
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att  # Broadcast multiplication
        
        # Spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att  # Broadcast multiplication
        
        return x


class CBAMBlock(nn.Module):
    """
    CBAM Block with residual connection
    Can be inserted into any part of the network
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        
        self.cbam = CBAM(in_channels, reduction_ratio, kernel_size)
        
    def forward(self, x):
        """
        Args:
            x: input features (B, C, H, W)
        Returns:
            refined features with residual connection (B, C, H, W)
        """
        residual = x
        out = self.cbam(x)
        out = out + residual  # Residual connection
        
        return out