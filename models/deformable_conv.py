"""
Deformable Convolution Modules (DCNv1, v2, v3, v4)
For adaptive feature extraction in camouflaged object detection
"""

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


# ============================================================================
# DCNv1 - Basic Deformable Convolution
# ============================================================================

class DCNv1Module(nn.Module):
    """
    DCNv1 - Deformable Convolution with learnable offsets
    Good for detecting weak/ambiguous boundaries
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Offset prediction network
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size,  # 2 * K * K (x, y offsets)
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        # Deformable convolution
        self.dcn = DeformConv2d(
            in_channels,
            out_channels, 
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize offset to zero (start with regular convolution)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
    def forward(self, x):
        # Predict offsets
        offset = self.offset_conv(x)
        
        # Apply deformable convolution
        out = self.dcn(x, offset)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


# ============================================================================
# DCNv2 - Deformable Convolution with Modulation
# ============================================================================

class DCNv2Module(nn.Module):
    """
    DCNv2 - Deformable Convolution with offsets + modulation weights
    Modulation helps suppress background noise - important for camouflage
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        K = kernel_size * kernel_size
        
        # Predict both offsets and modulation weights
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * K,  # 2*K offsets + K modulation weights
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size, 
            padding=padding,
            dilation=dilation
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)
        
    def forward(self, x):
        # Predict offsets and modulation weights
        out = self.offset_mask_conv(x)
        K = self.dcn.kernel_size[0] * self.dcn.kernel_size[1]
        
        # Split into offsets and modulation
        offset = out[:, :2*K, :, :]
        mask = torch.sigmoid(out[:, 2*K:, :, :])  # Modulation in [0,1]
        
        # Apply modulated deformable convolution
        out = self.dcn(x, offset, mask)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


# ============================================================================
# DCNv3 - Group-wise Deformable Convolution
# ============================================================================

class DCNv3Module(nn.Module):
    """
    DCNv3 - Group-wise deformable convolution
    Each group learns different deformation patterns
    Good for multi-texture camouflage
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=4):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Auto-adjust num_groups to ensure divisibility
        while in_channels % num_groups != 0 or out_channels % num_groups != 0:
            num_groups = max(1, num_groups - 1)
        
        self.num_groups = num_groups
        K = kernel_size * kernel_size
        
        # Group-wise offset and modulation prediction
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            num_groups * 3 * K,  # Per group: 2*K offsets + K masks
            kernel_size=1  # 1x1 conv for efficiency
        )
        
        # Separate DCN for each group
        self.group_dcns = nn.ModuleList([
            DeformConv2d(
                in_channels // num_groups,
                out_channels // num_groups,
                kernel_size=kernel_size,
                padding=padding
            ) for _ in range(num_groups)
        ])
        
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.GELU()
        
        # Offset normalization to prevent too large deformations
        self.offset_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Predict group-wise offsets and masks
        offset_mask = self.offset_mask_conv(x)
        
        outputs = []
        ch_per_group = C // self.num_groups
        K = self.kernel_size * self.kernel_size
        
        # Process each group
        for g in range(self.num_groups):
            # Extract group features
            x_g = x[:, g*ch_per_group:(g+1)*ch_per_group, :, :]
            
            # Extract group-specific offsets and masks
            start_idx = g * 3 * K
            offset_g = offset_mask[:, start_idx:start_idx+2*K, :, :]
            mask_g = torch.sigmoid(offset_mask[:, start_idx+2*K:start_idx+3*K, :, :])
            
            # Normalize offsets
            offset_g = offset_g * self.offset_scale
            
            # Apply group deformable conv
            out_g = self.group_dcns[g](x_g, offset_g, mask_g)
            outputs.append(out_g)
        
        # Concatenate group outputs
        out = torch.cat(outputs, dim=1)
        out = self.norm(out)
        out = self.activation(out)
        
        return out


# ============================================================================
# DCNv4 - Multi-scale Dynamic Deformable Convolution
# ============================================================================

class DCNv4Module(nn.Module):
    """
    DCNv4 - Multi-scale dynamic deformable convolution
    Combines multiple kernel sizes - critical for varying camouflage scales
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        
        # Auto-adjust num_groups
        while in_channels % num_groups != 0 or out_channels % num_groups != 0:
            num_groups = max(1, num_groups - 1)
        
        self.num_groups = num_groups
        
        # Smart channel distribution across scales
        out_per_scale = out_channels // 3
        remainder = out_channels % 3
        
        out_small = out_per_scale + (1 if remainder > 0 else 0)
        out_medium = out_per_scale + (1 if remainder > 1 else 0)
        out_large = out_channels - out_small - out_medium
        
        # Multi-scale DCNv3 modules
        self.dcn_small = DCNv3Module(
            in_channels, out_small, 
            kernel_size=3, padding=1, 
            num_groups=max(1, num_groups//2)
        )
        self.dcn_medium = DCNv3Module(
            in_channels, out_medium, 
            kernel_size=5, padding=2, 
            num_groups=max(1, num_groups//2)
        )
        self.dcn_large = DCNv3Module(
            in_channels, out_large, 
            kernel_size=7, padding=3, 
            num_groups=max(1, num_groups//2)
        )
        
        # Scale attention - learn which scale is important
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Compute scale attention weights
        scale_weights = self.scale_attention(x)  # [B, 3, 1, 1]
        
        # Multi-scale deformable features
        feat_s = self.dcn_small(x)
        feat_m = self.dcn_medium(x)
        feat_l = self.dcn_large(x)
        
        # Apply scale-specific weighting
        feat_s = feat_s * scale_weights[:, 0:1, :, :]
        feat_m = feat_m * scale_weights[:, 1:2, :, :]
        feat_l = feat_l * scale_weights[:, 2:3, :, :]
        
        # Concatenate and fuse
        out = torch.cat([feat_s, feat_m, feat_l], dim=1)
        out = self.fusion(out)
        
        return out