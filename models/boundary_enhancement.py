"""
Boundary Enhancement Module for Camouflaged Object Detection
Uses Sobel filters to enhance weak boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryEnhancementModule(nn.Module):
    """
    Enhance weak boundaries using edge-aware features
    Uses Sobel filters for boundary detection
    Can optionally predict boundary map for boundary loss
    """
    def __init__(self, channels, predict_boundary=False):
        super().__init__()
        
        self.predict_boundary = predict_boundary
        
        # Edge detection branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Sobel filters for boundary detection (fixed, not learned)
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]).float().view(1, 1, 3, 3))
        
        # Fusion of original and edge-enhanced features
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        
        # Optional boundary prediction head
        if predict_boundary:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(channels, channels // 2, 3, padding=1),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, 1, 1)  # Predict boundary map
            )
        
    def forward(self, x, return_boundary=False):
        """
        Args:
            x: input features (B, C, H, W)
            return_boundary: whether to return boundary prediction
        Returns:
            If return_boundary=False: enhanced features only
            If return_boundary=True: (enhanced_features, boundary_pred)
        """
        # Extract edge features
        edge_feat = self.edge_conv(x)
        
        # Combine original and edge-enhanced features
        out = torch.cat([x, edge_feat], dim=1)
        out = self.fusion(out)
        
        # Optionally predict boundary
        if return_boundary and self.predict_boundary:
            boundary_pred = self.boundary_head(out)
            return out, boundary_pred
        else:
            return out
    
    def extract_boundary_map(self, x):
        """
        Explicitly extract boundary map using Sobel filters
        Useful for creating ground truth boundary maps
        Args:
            x: input features (B, C, H, W) or (B, 1, H, W)
        Returns:
            boundary magnitude map (B, 1, H, W)
        """
        # Average across channels if multi-channel
        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x
        
        # Apply Sobel filters
        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        
        # Gradient magnitude
        boundary = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Normalize to [0, 1]
        boundary = boundary / (boundary.max() + 1e-8)  # OR
        boundary = torch.clamp(boundary, 0, 1)
        
        return boundary



class BEM_Multilevel(nn.Module):
    """
    Multi-level Boundary Enhancement Module (Edge Exploration Module)
    
    Based on the Edge Exploration Module architecture:
    - Takes multi-level decoder features (f1, f2, f3, f4)
    - Upsamples all features to the same resolution (f1's resolution)
    - Concatenates to form fe'
    - Applies multi-scale dilated convolutions (dilation=1, 3, 5)
    - Uses Retrieve Attention mechanism
    - Outputs enhanced edge features
    
    Architecture:
        f1 ────────────────┐
        f2 ──► Upsample ──►│
        f3 ──► Upsample ──►├──► Concat ──► fe' ──► [Dilated Conv d=1] ──► d1 ─┐
        f4 ──► Upsample ──►│                   ──► [Dilated Conv d=3] ──► d2 ─┼─► Concat ──► d ──► Attention ──► fe
                                               ──► [Dilated Conv d=5] ──► d3 ─┘
    
    Args:
        in_channels_list: List of input channels for each level [c1, c2, c3, c4]
                         If single int, assumes same channels for all levels
        out_channels: Output channels (default: 64)
        predict_boundary: Whether to predict boundary map (default: False)
    """
    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()       
        
        # Handle single channel input (for compatibility with BoundaryEnhancementModule)
        if isinstance(in_channels_list, int):
            # Single input mode - same as original BEM
            self.single_input_mode = True
            in_channels = in_channels_list
            self.in_channels_list = [in_channels]
        else:
            self.single_input_mode = False
            self.in_channels_list = in_channels_list
        
        # ===================== Feature Alignment =====================
        # Conv layers to align all features to same channel dimension
        if not self.single_input_mode:
            self.align_convs = nn.ModuleList()
            for in_ch in self.in_channels_list:
                self.align_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            
            # After concatenation: out_channels * num_levels
            concat_channels = out_channels * len(self.in_channels_list)
        else:
            concat_channels = in_channels_list  # Single input
        
        # ===================== Feature Fusion (fe') =====================
        # Reduce concatenated features
        self.fe_prime = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ===================== Multi-scale Dilated Convolutions =====================
        # d1: dilation = 1 (standard convolution)
        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # d2: dilation = 3
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # d3: dilation = 5
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ===================== Feature Aggregation (d) =====================
        # Fuse multi-scale features: d = concat(d1, d2, d3)
        self.aggregation = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ===================== Retrieve Attention =====================
        # Attention mechanism to focus on edge regions
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ===================== Output Refinement =====================
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ===================== Boundary Prediction Head =====================
        self.boundary_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 1, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Can be either:
               - Single tensor (B, C, H, W) for single input mode
               - List of tensors [f1, f2, f3, f4] for multi-level mode
                 where f1 is the highest resolution (from decoder1)
            return_boundary: Whether to return boundary prediction
        
        Returns:
            If return_boundary=False: 
                enhanced features (B, out_channels, H, W)
            If return_boundary=True: 
                (enhanced_features, boundary_pred)
        """
        # ===================== Handle Input =====================
        if self.single_input_mode:
            # Single input mode - compatible with original BEM
            if isinstance(x, list):
                x = x[0]
            target_size = x.shape[2:]
            fe_prime = self.fe_prime(x)
        else:
            # Multi-level input mode
            if not isinstance(x, (list, tuple)):
                raise ValueError("Multi-level mode expects a list of feature maps [f1, f2, f3, f4]")
            
            # Target size is f1's resolution (highest resolution)
            target_size = x[0].shape[2:]
            
            # Align channels and upsample to target size
            aligned_features = []
            for i, (feat, conv) in enumerate(zip(x, self.align_convs)):
                # Align channels
                feat = conv(feat)
                
                # Upsample to target size if needed
                if feat.shape[2:] != target_size:
                    feat = F.interpolate(
                        feat, 
                        size=target_size, 
                        mode='bilinear', 
                        align_corners=True
                    )
                aligned_features.append(feat)
            
            # Concatenate all aligned features
            concat_feat = torch.cat(aligned_features, dim=1)
            
            # Generate fe' (fused edge feature)
            fe_prime = self.fe_prime(concat_feat)
        
        # ===================== Multi-scale Dilated Convolutions =====================
        # Apply dilated convolutions with different dilation rates
        d1 = self.dilated_conv1(fe_prime)  # dilation = 1
        d2 = self.dilated_conv2(fe_prime)  # dilation = 3
        d3 = self.dilated_conv3(fe_prime)  # dilation = 5
        
        # Concatenate multi-scale features
        d = torch.cat([d1, d2, d3], dim=1)
        
        # Aggregate multi-scale features
        d = self.aggregation(d)
        
        # ===================== Retrieve Attention =====================
        # Generate attention map
        attn = self.attention(d)
        
        # Apply attention (element-wise multiplication)
        fe = d * attn
        
        # ===================== Output Refinement =====================
        fe = self.output_conv(fe)
        
        # ===================== Boundary Prediction =====================
        boundary_pred = self.boundary_head(fe)
        return fe, boundary_pred
    
    def extract_boundary_map(self, x):
        """
        Explicitly extract boundary map using Sobel filters
        Useful for creating ground truth boundary maps
        
        Args:
            x: input features (B, C, H, W) or (B, 1, H, W)
        
        Returns:
            boundary magnitude map (B, 1, H, W)
        """
        # Sobel filters
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3).to(x.device)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]).float().view(1, 1, 3, 3).to(x.device)
        
        # Average across channels if multi-channel
        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x
        
        # Apply Sobel filters
        grad_x = F.conv2d(x_gray, sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, sobel_y, padding=1)
        
        # Gradient magnitude
        boundary = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Normalize to [0, 1]
        boundary = boundary / (boundary.max() + 1e-8)
        boundary = torch.clamp(boundary, 0, 1)
        
        return boundary