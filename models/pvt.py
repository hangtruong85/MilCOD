import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .boundary_enhancement import BoundaryEnhancementModule, BEM_Multilevel
from .cbam import CBAM


# ============================================================================
# Decoder Block
# ============================================================================
"""
Decoder Block for full-scale skip connections
Aggregates features from ALL encoder levels
"""
class DecoderBlock(nn.Module):
    """
    Decoder Block with full-scale skip connections
    Aggregates features from ALL encoder levels
    
    Args:
        in_channels_list: list of input channels from each encoder level
        out_channels: output channels after fusion
    
    Example:
        >>> decoder4 = DecoderBlock(
        ...     in_channels_list=[64, 128, 320, 512],
        ...     out_channels=64
        ... )
    """
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list: list of input channels from each encoder level
            out_channels: output channels after fusion
        """
        super().__init__()
        
        # Calculate channels after concatenation
        cat_channels = out_channels * len(in_channels_list)
        
        # Convs to reduce each encoder feature to out_channels
        self.conv_branches = nn.ModuleList()
        for in_ch in in_channels_list:
            self.conv_branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Fusion after concatenation
        self.fusion = nn.Sequential(
            nn.Conv2d(cat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, target_size):
        """
        Args:
            features: list of feature maps from all encoder levels
            target_size: (H, W) target size for this decoder level
        
        Returns:
            out: fused feature map of shape (B, out_channels, H, W)
        """
        processed = []
        
        for feat, conv in zip(features, self.conv_branches):
            # Resize to target size if needed
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=True
                )
            
            # Process with conv
            feat = conv(feat)
            processed.append(feat)
        
        # Concatenate and fuse
        cat_feat = torch.cat(processed, dim=1)
        out = self.fusion(cat_feat)
        
        return out

class PVT(nn.Module):
    """
    a Net with PVT-V2-B2 encoder
    Full-scale skip connections
    
    Args:
        n_classes: Number of output classes (default: 1)
    """
    def __init__(self, n_classes=1):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder 4: aggregate from e1, e2, e3, e4
        self.decoder4 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        # Decoder 3: aggregate from e1, e2, e3, d4
        self.decoder3 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 2: aggregate from e1, e2, d3, d4
        self.decoder2 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 1: aggregate from e1, d2, d3, d4
        self.decoder1 = DecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Calculate target sizes
        size_d4 = (features[3].shape[2], features[3].shape[3])
        size_d3 = (features[2].shape[2], features[2].shape[3])
        size_d2 = (features[1].shape[2], features[1].shape[3])
        size_d1 = (features[0].shape[2], features[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features[0], features[1], features[2], features[3]], size_d4)
        d3 = self.decoder3([features[0], features[1], features[2], d4], size_d3)
        d2 = self.decoder2([features[0], features[1], d3, d4], size_d2)
        d1 = self.decoder1([features[0], d2, d3, d4], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation
        mask = self.segmentation_head(d1)
        
        return mask


class PVT_BEM(nn.Module):
    """
    UNet3+ with PVT-V2-B2 encoder + Boundary Enhancement Module
    Full-scale skip connections with boundary prediction
    
    Args:
        n_classes: Number of output classes (default: 1)
        predict_boundary: Whether to predict boundary (default: True)
    """
    def __init__(self, n_classes=1, predict_boundary=True):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder 4: aggregate from e1, e2, e3, e4
        self.decoder4 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        # Decoder 3: aggregate from e1, e2, e3, d4
        self.decoder3 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 2: aggregate from e1, e2, d3, d4
        self.decoder2 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 1: aggregate from e1, d2, d3, d4
        self.decoder1 = DecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
        # Boundary Enhancement Module
        self.predict_boundary = predict_boundary
        self.bem = BoundaryEnhancementModule(decoder_channels, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Calculate target sizes
        size_d4 = (features[3].shape[2], features[3].shape[3])
        size_d3 = (features[2].shape[2], features[2].shape[3])
        size_d2 = (features[1].shape[2], features[1].shape[3])
        size_d1 = (features[0].shape[2], features[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features[0], features[1], features[2], features[3]], size_d4)
        d3 = self.decoder3([features[0], features[1], features[2], d4], size_d3)
        d2 = self.decoder2([features[0], features[1], d3, d4], size_d2)
        d1 = self.decoder1([features[0], d2, d3, d4], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Boundary enhancement
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(d1, return_boundary=True)
        else:
            decoder_output = self.bem(d1, return_boundary=False)
        
        # Segmentation
        mask = self.segmentation_head(decoder_output)
        
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask


class PVT_CBAM(nn.Module):
    """
    a Net with PVT-V2-B2 encoder + CBAM on encoder levels 3 & 4
    Full-scale skip connections with channel and spatial attention
    
    Args:
        n_classes: Number of output classes (default: 1)
    """
    def __init__(self, n_classes=1):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # CBAM modules for encoder levels 3 & 4
        self.cbam_e3 = CBAM(encoder_channels[3])  # 320 channels
        self.cbam_e4 = CBAM(encoder_channels[4])  # 512 channels
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder 4: aggregate from e1, e2, e3, e4
        self.decoder4 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        # Decoder 3: aggregate from e1, e2, e3, d4
        self.decoder3 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 2: aggregate from e1, e2, d3, d4
        self.decoder2 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 1: aggregate from e1, d2, d3, d4
        self.decoder1 = DecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Apply CBAM to encoder levels 3 & 4
        features_with_cbam = [
            features[0],                    # e1 (no CBAM)
            features[1],                    # e2 (no CBAM)
            self.cbam_e3(features[2]),      # e3 (with CBAM)
            self.cbam_e4(features[3])       # e4 (with CBAM)
        ]
        
        # Calculate target sizes
        size_d4 = (features_with_cbam[3].shape[2], features_with_cbam[3].shape[3])
        size_d3 = (features_with_cbam[2].shape[2], features_with_cbam[2].shape[3])
        size_d2 = (features_with_cbam[1].shape[2], features_with_cbam[1].shape[3])
        size_d1 = (features_with_cbam[0].shape[2], features_with_cbam[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features_with_cbam[0], features_with_cbam[1], features_with_cbam[2], features_with_cbam[3]], size_d4)
        d3 = self.decoder3([features_with_cbam[0], features_with_cbam[1], features_with_cbam[2], d4], size_d3)
        d2 = self.decoder2([features_with_cbam[0], features_with_cbam[1], d3, d4], size_d2)
        d1 = self.decoder1([features_with_cbam[0], d2, d3, d4], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation
        mask = self.segmentation_head(d1)
        
        return mask


class PVT_BEM_CBAM(nn.Module):
    """
    a Net with PVT-V2-B2 encoder + BEM + CBAM on encoder levels 3 & 4
    Full-scale skip connections with boundary prediction and attention
    
    Args:
        n_classes: Number of output classes (default: 1)
        predict_boundary: Whether to predict boundary (default: True)
    """
    def __init__(self, n_classes=1, predict_boundary=True):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # CBAM modules for encoder levels 3 & 4
        self.cbam_e3 = CBAM(encoder_channels[3])  # 320 channels
        self.cbam_e4 = CBAM(encoder_channels[4])  # 512 channels
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder 4: aggregate from e1, e2, e3, e4
        self.decoder4 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        # Decoder 3: aggregate from e1, e2, e3, d4
        self.decoder3 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 2: aggregate from e1, e2, d3, d4
        self.decoder2 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Decoder 1: aggregate from e1, d2, d3, d4
        self.decoder1 = DecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels, n_classes, 1)
        
        # Boundary Enhancement Module
        self.predict_boundary = predict_boundary
        self.bem = BoundaryEnhancementModule(decoder_channels, predict_boundary=predict_boundary)
        
    def forward(self, x, return_boundary=False):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Apply CBAM to encoder levels 3 & 4
        features_with_cbam = [
            features[0],                    # e1 (no CBAM)
            features[1],                    # e2 (no CBAM)
            self.cbam_e3(features[2]),      # e3 (with CBAM)
            self.cbam_e4(features[3])       # e4 (with CBAM)
        ]
        
        # Calculate target sizes
        size_d4 = (features_with_cbam[3].shape[2], features_with_cbam[3].shape[3])
        size_d3 = (features_with_cbam[2].shape[2], features_with_cbam[2].shape[3])
        size_d2 = (features_with_cbam[1].shape[2], features_with_cbam[1].shape[3])
        size_d1 = (features_with_cbam[0].shape[2], features_with_cbam[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features_with_cbam[0], features_with_cbam[1], features_with_cbam[2], features_with_cbam[3]], size_d4)
        d3 = self.decoder3([features_with_cbam[0], features_with_cbam[1], features_with_cbam[2], d4], size_d3)
        d2 = self.decoder2([features_with_cbam[0], features_with_cbam[1], d3, d4], size_d2)
        d1 = self.decoder1([features_with_cbam[0], d2, d3, d4], size_d1)
        
        # Upsample to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Boundary enhancement
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(d1, return_boundary=True)
        else:
            decoder_output = self.bem(d1, return_boundary=False)
        
        # Segmentation
        mask = self.segmentation_head(decoder_output)
        
        if return_boundary and self.predict_boundary:
            return mask, boundary_pred
        else:
            return mask

"""
Deep Supervision Implementation for Segmentation
- Multiple outputs from decoder levels (d1, d2, d3, d4)
"""

# ============================================================================
# Deep Supervision Models
# ============================================================================

class DeepSupervision(nn.Module):
    """
    Base class for a Net with Deep Supervision
    Returns predictions from all decoder levels for deep supervision
    
    This is a template - inherit and implement forward() specific to your encoder
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Should return: (d1, d2, d3, d4) predictions
        Or: ((d1, d2, d3, d4), boundary) if with boundary
        """
        raise NotImplementedError("Implement forward() in subclass")


def add_deep_supervision_heads(model_class):
    """
    Decorator to add deep supervision heads to existing decoder blocks
    
    Usage:
        @add_deep_supervision_heads
        class MyUNet3Plus(nn.Module):
            ...
    """
    def wrapper(*args, **kwargs):
        instance = model_class(*args, **kwargs)
        
        # If model has segmentation_head, modify to output at all levels
        if hasattr(instance, 'decoder1'):
            # Add output heads for each decoder level
            decoder_channels = 64  # Default
            
            # d1, d2, d3, d4 output heads
            instance.out_d1 = nn.Conv2d(decoder_channels, 1, 1)
            instance.out_d2 = nn.Conv2d(decoder_channels, 1, 1)
            instance.out_d3 = nn.Conv2d(decoder_channels, 1, 1)
            instance.out_d4 = nn.Conv2d(decoder_channels, 1, 1)
        
        return instance
    
    return wrapper


# ============================================================================
# Example Integration with Existing Models
# ============================================================================

class PVT_DeepSupervision(nn.Module):
    """
    a Net with PVT-V2-B2 encoder + Deep Supervision + BEM + CBAM
    
    Args:
        n_classes: Number of output classes (default: 1)
        predict_boundary: Whether to predict boundary (default: True)
    """
    def __init__(self, n_classes=1, predict_boundary=True):
        super().__init__()
        
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # CBAM modules for encoder levels 3 & 4
        self.cbam_e3 = CBAM(encoder_channels[3])  # 320 channels
        self.cbam_e4 = CBAM(encoder_channels[4])  # 512 channels
        
        # Decoder channels
        decoder_channels = 64
        
        # Decoder blocks
        self.decoder4 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        self.decoder3 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        self.decoder2 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        self.decoder1 = DecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        # ================== DEEP SUPERVISION HEADS ==================
        # Output heads for each decoder level
        self.out_d4 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d3 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d2 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d1 = nn.Conv2d(decoder_channels, n_classes, 1)
        
        # Boundary Enhancement Module
        self.predict_boundary = predict_boundary
        self.bem = BoundaryEnhancementModule(decoder_channels, predict_boundary=predict_boundary)
    
    def forward(self, x, return_all_levels=False, return_boundary=False):
        """
        Forward pass with deep supervision
        
        Args:
            x: Input image (B, 3, H, W)
            return_all_levels: Whether to return predictions from all decoder levels
                              If True, returns (d1, d2, d3, d4)
                              If False, returns only d1 (final prediction)
            return_boundary: Whether to return boundary prediction
        
        Returns:
            If return_all_levels=False:
                mask: Final segmentation (B, 1, H, W)
                or (mask, boundary) if return_boundary=True
            
            If return_all_levels=True:
                (pred_d1, pred_d2, pred_d3, pred_d4): All level predictions
                or ((pred_d1, pred_d2, pred_d3, pred_d4), boundary) if return_boundary=True
        """
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Apply CBAM to encoder levels 3 & 4
        features_with_cbam = [
            features[0],                    # e1 (no CBAM)
            features[1],                    # e2 (no CBAM)
            self.cbam_e3(features[2]),      # e3 (with CBAM)
            self.cbam_e4(features[3])       # e4 (with CBAM)
        ]
        
        # Calculate target sizes
        size_d4 = (features_with_cbam[3].shape[2], features_with_cbam[3].shape[3])
        size_d3 = (features_with_cbam[2].shape[2], features_with_cbam[2].shape[3])
        size_d2 = (features_with_cbam[1].shape[2], features_with_cbam[1].shape[3])
        size_d1 = (features_with_cbam[0].shape[2], features_with_cbam[0].shape[3])
        
        # Decoder
        d4 = self.decoder4([features_with_cbam[0], features_with_cbam[1], 
                           features_with_cbam[2], features_with_cbam[3]], size_d4)
        d3 = self.decoder3([features_with_cbam[0], features_with_cbam[1], 
                           features_with_cbam[2], d4], size_d3)
        d2 = self.decoder2([features_with_cbam[0], features_with_cbam[1], 
                           d3, d4], size_d2)
        d1 = self.decoder1([features_with_cbam[0], d2, d3, d4], size_d1)
        
        # Upsample d1 to input size
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # Boundary enhancement
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(d1, return_boundary=True)
        else:
            decoder_output = self.bem(d1, return_boundary=False)
            boundary_pred = None
        
        # ================== DEEP SUPERVISION OUTPUTS ==================
        # Output predictions from all levels
        pred_d4 = self.out_d4(d4)
        pred_d3 = self.out_d3(d3)
        pred_d2 = self.out_d2(d2)
        pred_d1 = self.out_d1(decoder_output)
        
        # Return based on flags
        if return_all_levels:
            if return_boundary and self.predict_boundary:
                return (pred_d1, pred_d2, pred_d3, pred_d4), boundary_pred
            else:
                return (pred_d1, pred_d2, pred_d3, pred_d4), None
        else:
            if return_boundary and self.predict_boundary:
                return pred_d1, boundary_pred
            else:
                return pred_d1, None

class PVT_DeepSupervision_BEMMulti(nn.Module):
    """
    PVT-V2-B2 encoder + Deep Supervision + BEM_Multilevel + CBAM
    
    Architecture:
    - Encoder: PVT-V2-B2 (4 stages)
      ├─ Stage 1: 64 channels
      ├─ Stage 2: 128 channels
      ├─ Stage 3: 320 channels ← CBAM applied
      └─ Stage 4: 512 channels ← CBAM applied
    
    - Decoder: 4 levels with full-scale skip connections
      ├─ d4: e1 + e2 + e3 + e4
      ├─ d3: e1 + e2 + e3 + d4
      ├─ d2: e1 + e2 + d3 + d4
      └─ d1: e1 + d2 + d3 + d4
    
    - BEM_Multilevel: Edge Exploration Module
      └─ Takes [d1, d2, d3, d4] as multi-level input
      └─ Multi-scale dilated convolutions (d=1, 3, 5)
      └─ Retrieve Attention mechanism
    
    - Deep Supervision: Output heads at all decoder levels
    
    Args:
        n_classes: Number of output classes (default: 1)
        predict_boundary: Whether to predict boundary (default: True)
    """
    def __init__(self, n_classes=1, predict_boundary=True):
        super().__init__()
        
        # ================== ENCODER ==================
        # Load PVT-V2-B2 from timm
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        
        # PVT-V2-B2 output channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]
        
        # ================== CBAM ==================
        # CBAM modules for encoder levels 3 & 4 (deep semantic features)
        self.cbam_e3 = CBAM(encoder_channels[3])  # 320 channels
        self.cbam_e4 = CBAM(encoder_channels[4])  # 512 channels
        
        # ================== DECODER ==================
        decoder_channels = 64
        
        # Decoder blocks with full-scale skip connections
        # d4 = e1 + e2 + e3 + e4
        self.decoder4 = DecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 64
                encoder_channels[2],  # e2: 128
                encoder_channels[3],  # e3: 320
                encoder_channels[4]   # e4: 512
            ],
            out_channels=decoder_channels
        )
        
        # d3 = e1 + e2 + e3 + d4
        self.decoder3 = DecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 64
                encoder_channels[2],  # e2: 128
                encoder_channels[3],  # e3: 320
                decoder_channels      # d4: 64
            ],
            out_channels=decoder_channels
        )
        
        # d2 = e1 + e2 + d3 + d4
        self.decoder2 = DecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 64
                encoder_channels[2],  # e2: 128
                decoder_channels,     # d3: 64
                decoder_channels      # d4: 64
            ],
            out_channels=decoder_channels
        )
        
        # d1 = e1 + d2 + d3 + d4
        self.decoder1 = DecoderBlock(
            in_channels_list=[
                encoder_channels[1],  # e1: 64
                decoder_channels,     # d2: 64
                decoder_channels,     # d3: 64
                decoder_channels      # d4: 64
            ],
            out_channels=decoder_channels
        )
        
        # ================== BEM_Multilevel ==================
        # Edge Exploration Module with multi-level input
        # Takes [d1, d2, d3, d4] as input
        self.predict_boundary = predict_boundary
        self.bem = BEM_Multilevel(
            in_channels_list=[decoder_channels, decoder_channels, decoder_channels, decoder_channels],
            out_channels=decoder_channels,
            predict_boundary=predict_boundary
        )
        
        # ================== DEEP SUPERVISION HEADS ==================
        # Output heads for each decoder level
        self.out_d4 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d3 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d2 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d1 = nn.Conv2d(decoder_channels, n_classes, 1)
    
    def forward(self, x, return_all_levels=False, return_boundary=False):
        """
        Forward pass with deep supervision
        
        Args:
            x: Input image (B, 3, H, W)
            return_all_levels: Whether to return predictions from all decoder levels
                              If True, returns (d1, d2, d3, d4)
                              If False, returns only d1 (final prediction)
            return_boundary: Whether to return boundary prediction
        
        Returns:
            If return_all_levels=False:
                mask: Final segmentation (B, 1, H, W)
                or (mask, boundary) if return_boundary=True
            
            If return_all_levels=True:
                (pred_d1, pred_d2, pred_d3, pred_d4): All level predictions
                or ((pred_d1, pred_d2, pred_d3, pred_d4), boundary) if return_boundary=True
        """
        input_size = x.shape[2:]
        
        # ================== ENCODER ==================
        features = self.backbone(x)  # [e1, e2, e3, e4]
        
        # Apply CBAM to encoder levels 3 & 4
        e1 = features[0]                    # e1 (no CBAM)
        e2 = features[1]                    # e2 (no CBAM)
        e3 = self.cbam_e3(features[2])      # e3 (with CBAM)
        e4 = self.cbam_e4(features[3])      # e4 (with CBAM)
        
        # ================== DECODER ==================
        # Calculate target sizes for each decoder level
        size_d4 = (e4.shape[2], e4.shape[3])
        size_d3 = (e3.shape[2], e3.shape[3])
        size_d2 = (e2.shape[2], e2.shape[3])
        size_d1 = (e1.shape[2], e1.shape[3])
        
        # Decoder forward
        # d4 = e1 + e2 + e3 + e4
        d4 = self.decoder4([e1, e2, e3, e4], size_d4)
        
        # d3 = e1 + e2 + e3 + d4
        d3 = self.decoder3([e1, e2, e3, d4], size_d3)
        
        # d2 = e1 + e2 + d3 + d4
        d2 = self.decoder2([e1, e2, d3, d4], size_d2)
        
        # d1 = e1 + d2 + d3 + d4
        d1 = self.decoder1([e1, d2, d3, d4], size_d1)
        
        # Upsample d1 to input size
        d1_upsampled = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        
        # ================== BEM_Multilevel ==================
        # Upsample all decoder features to input size for BEM
        d2_upsampled = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)
        d3_upsampled = F.interpolate(d3, size=input_size, mode='bilinear', align_corners=True)
        d4_upsampled = F.interpolate(d4, size=input_size, mode='bilinear', align_corners=True)
        
        # BEM takes multi-level decoder features [d1, d2, d3, d4]
        # d1 is highest resolution, d4 is lowest
        if return_boundary and self.predict_boundary:
            decoder_output, boundary_pred = self.bem(
                [d1_upsampled, d2_upsampled, d3_upsampled, d4_upsampled], 
                return_boundary=True
            )
        else:
            decoder_output = self.bem(
                [d1_upsampled, d2_upsampled, d3_upsampled, d4_upsampled], 
                return_boundary=False
            )
            boundary_pred = None
        
        # ================== DEEP SUPERVISION OUTPUTS ==================
        # Output predictions from all levels
        pred_d4 = self.out_d4(d4)
        pred_d3 = self.out_d3(d3)
        pred_d2 = self.out_d2(d2)
        pred_d1 = self.out_d1(decoder_output)  # Final output after BEM
        
        # ================== RETURN ==================
        if return_all_levels:
            if return_boundary and self.predict_boundary:
                return (pred_d1, pred_d2, pred_d3, pred_d4), boundary_pred
            else:
                return (pred_d1, pred_d2, pred_d3, pred_d4), None
        else:
            if return_boundary and self.predict_boundary:
                return pred_d1, boundary_pred
            else:
                return pred_d1, None
