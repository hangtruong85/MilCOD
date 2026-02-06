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

# ==================== BI-DIRECTIONAL FUSION MODULE (BFM) ====================
class BiDirectionalFusionModule(nn.Module):
    """
    Bi-directional Fusion Module (BFM)
    
    Fuses RGB and Depth features using bi-directional attention:
    - Depth-to-RGB (D2R): Depth guides RGB to focus on spatial structure changes
    - RGB-to-Depth (R2D): RGB helps filter geometric noise in depth
    
    Architecture:
        F_rgb ──┬──► D2R Attention ──► F_rgb' ──┐
                │                                 ├──► Concat ──► Conv ──► F_fusion
        F_d ────┴──► R2D Attention ──► F_d' ────┘
    
    Args:
        rgb_channels: Number of channels in RGB features
        depth_channels: Number of channels in Depth features
        out_channels: Number of output channels (default: same as rgb_channels)
    """
    def __init__(self, rgb_channels, depth_channels, out_channels=None):
        super().__init__()
        
        if out_channels is None:
            out_channels = rgb_channels
        
        # ========== Channel Alignment ==========
        # Align depth channels to match RGB if different
        if depth_channels != rgb_channels:
            self.align_depth = nn.Sequential(
                nn.Conv2d(depth_channels, rgb_channels, kernel_size=1),
                nn.BatchNorm2d(rgb_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.align_depth = nn.Identity()
        
        # ========== Depth-to-RGB (D2R) Attention ==========
        # Depth guides RGB to focus on structural boundaries
        self.d2r_query = nn.Conv2d(rgb_channels, rgb_channels // 8, kernel_size=1)
        self.d2r_key = nn.Conv2d(rgb_channels, rgb_channels // 8, kernel_size=1)
        self.d2r_value = nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1)
        self.d2r_gamma = nn.Parameter(torch.zeros(1))
        
        # ========== RGB-to-Depth (R2D) Attention ==========
        # RGB helps filter depth noise
        self.r2d_query = nn.Conv2d(rgb_channels, rgb_channels // 8, kernel_size=1)
        self.r2d_key = nn.Conv2d(rgb_channels, rgb_channels // 8, kernel_size=1)
        self.r2d_value = nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1)
        self.r2d_gamma = nn.Parameter(torch.zeros(1))
        
        # ========== Fusion ==========
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, f_rgb, f_depth):
        """
        Args:
            f_rgb: RGB features (B, C_rgb, H, W)
            f_depth: Depth features (B, C_depth, H, W)
        
        Returns:
            f_fusion: Fused features (B, out_channels, H, W)
        """
        B, C_rgb, H, W = f_rgb.shape
        
        # Align depth channels
        f_depth_aligned = self.align_depth(f_depth)
        
        # Ensure spatial dimensions match
        if f_depth_aligned.shape[2:] != f_rgb.shape[2:]:
            f_depth_aligned = F.interpolate(
                f_depth_aligned, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=True
            )
        
        # ========== D2R: Depth-to-RGB Attention ==========
        # Query from RGB, Key/Value from Depth
        Q_d2r = self.d2r_query(f_rgb).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        K_d2r = self.d2r_key(f_depth_aligned).view(B, -1, H * W)  # (B, C', HW)
        V_d2r = self.d2r_value(f_depth_aligned).view(B, -1, H * W)  # (B, C, HW)
        
        # Attention: softmax(Q*K) * V
        attention_d2r = torch.softmax(torch.bmm(Q_d2r, K_d2r), dim=-1)  # (B, HW, HW)
        out_d2r = torch.bmm(V_d2r, attention_d2r.permute(0, 2, 1))  # (B, C, HW)
        out_d2r = out_d2r.view(B, C_rgb, H, W)
        
        # Residual connection with learnable weight
        f_rgb_enhanced = f_rgb + self.d2r_gamma * out_d2r
        
        # ========== R2D: RGB-to-Depth Attention ==========
        # Query from Depth, Key/Value from RGB
        Q_r2d = self.r2d_query(f_depth_aligned).view(B, -1, H * W).permute(0, 2, 1)
        K_r2d = self.r2d_key(f_rgb).view(B, -1, H * W)
        V_r2d = self.r2d_value(f_rgb).view(B, -1, H * W)
        
        attention_r2d = torch.softmax(torch.bmm(Q_r2d, K_r2d), dim=-1)
        out_r2d = torch.bmm(V_r2d, attention_r2d.permute(0, 2, 1))
        out_r2d = out_r2d.view(B, C_rgb, H, W)
        
        f_depth_enhanced = f_depth_aligned + self.r2d_gamma * out_r2d
        
        # ========== Fusion ==========
        # Concatenate enhanced features and fuse
        f_fusion = torch.cat([f_rgb_enhanced, f_depth_enhanced], dim=1)
        f_fusion = self.fusion_conv(f_fusion)
        
        return f_fusion


# ==================== DUAL-STREAM ENCODER WITH BFM ====================
class DualStreamEncoder(nn.Module):
    """
    Dual-Stream Encoder with Bi-directional Fusion
    
    Architecture:
        RGB Stream (PVT-v2-b2):  3ch → [64, 128, 320, 512]
        Depth Stream (PVT-v2-b0): 1ch → [32, 64, 160, 256]
        
        Stage 1: F1_rgb (64) + F1_d (32) → BFM → F1_fusion (64)
        Stage 2: F2_rgb (128) + F2_d (64) → BFM → F2_fusion (128)
        Stage 3: F3_rgb (320) + F3_d (160) → BFM → F3_fusion (320)
        Stage 4: F4_rgb (512) + F4_d (256) → BFM → F4_fusion (512)
    
    Args:
        pretrained: Whether to use pretrained weights
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # ========== RGB Stream (PVT-v2-b2) ==========
        self.rgb_backbone = timm.create_model(
            'pvt_v2_b2', 
            pretrained=pretrained, 
            features_only=True
        )
        # Output channels: [64, 128, 320, 512]
        self.rgb_channels = [64, 128, 320, 512]
        
        # ========== Depth Stream (PVT-v2-b0) ==========
        # Need to modify first conv to accept 1 channel instead of 3
        self.depth_backbone = timm.create_model(
            'pvt_v2_b0', 
            pretrained=pretrained, 
            features_only=True,
            in_chans=1  # Single channel for depth
        )
        # Output channels: [32, 64, 160, 256]
        self.depth_channels = [32, 64, 160, 256]
        
        # ========== Bi-directional Fusion Modules (BFM) ==========
        # One BFM for each stage
        self.bfm1 = BiDirectionalFusionModule(
            rgb_channels=self.rgb_channels[0],  # 64
            depth_channels=self.depth_channels[0],  # 32
            out_channels=self.rgb_channels[0]  # 64
        )
        
        self.bfm2 = BiDirectionalFusionModule(
            rgb_channels=self.rgb_channels[1],  # 128
            depth_channels=self.depth_channels[1],  # 64
            out_channels=self.rgb_channels[1]  # 128
        )
        
        self.bfm3 = BiDirectionalFusionModule(
            rgb_channels=self.rgb_channels[2],  # 320
            depth_channels=self.depth_channels[2],  # 160
            out_channels=self.rgb_channels[2]  # 320
        )
        
        self.bfm4 = BiDirectionalFusionModule(
            rgb_channels=self.rgb_channels[3],  # 512
            depth_channels=self.depth_channels[3],  # 256
            out_channels=self.rgb_channels[3]  # 512
        )
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB image (B, 3, H, W)
            depth: Depth map (B, 1, H, W)
        
        Returns:
            fused_features: List of fused features [F1, F2, F3, F4]
        """
        # ========== Extract Features ==========
        # RGB stream
        rgb_features = self.rgb_backbone(rgb)  # [F1_rgb, F2_rgb, F3_rgb, F4_rgb]
        
        # Depth stream
        depth_features = self.depth_backbone(depth)  # [F1_d, F2_d, F3_d, F4_d]
        
        # ========== Bi-directional Fusion at Each Stage ==========
        f1_fusion = self.bfm1(rgb_features[0], depth_features[0])
        f2_fusion = self.bfm2(rgb_features[1], depth_features[1])
        f3_fusion = self.bfm3(rgb_features[2], depth_features[2])
        f4_fusion = self.bfm4(rgb_features[3], depth_features[3])
        
        return [f1_fusion, f2_fusion, f3_fusion, f4_fusion]


# ==================== COMPLETE MODEL ====================
class DualStream_PVT_COD(nn.Module):
    """
    Dual-Stream PVT with Deep Supervision and BEM
    
    Architecture:
        Input: RGB (3ch) + Depth (1ch)
        
        Encoder:
            - RGB Stream: PVT-v2-b2 → [64, 128, 320, 512]
            - Depth Stream: PVT-v2-b0 → [32, 64, 160, 256]
            - BFM at each stage → [64, 128, 320, 512]
        
        Decoder:
            - Full-scale skip connections
            - CBAM on stages 3 & 4
            - BEM_Multilevel for edge enhancement
            - Deep supervision outputs
    
    Args:
        n_classes: Number of output classes
        is_bem: Whether to use BEM
        is_cbam_en3: Whether to use CBAM on encoder stage 3
        is_cbam_en4: Whether to use CBAM on encoder stage 4
        pretrained: Whether to use pretrained backbones
    """
    def __init__(self, n_classes=1, is_bem=True, is_cbam_en3=True, 
                 is_cbam_en4=True, pretrained=True):
        super().__init__()
        
        # ================== DUAL-STREAM ENCODER ==================
        self.dual_encoder = DualStreamEncoder(pretrained=pretrained)
        
        # Fused encoder channels: [64, 128, 320, 512]
        encoder_channels = [3, 64, 128, 320, 512]  # 3 for compatibility  
        
        # ================== FLAGS ==================
        self.is_bem = is_bem
        self.is_cbam_en3 = is_cbam_en3
        self.is_cbam_en4 = is_cbam_en4
        
        # ================== DECODER ==================
        decoder_channels = 64
        
        self.decoder4 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], 
                            encoder_channels[3], encoder_channels[4]],
            out_channels=decoder_channels
        )
        
        self.decoder3 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], 
                            encoder_channels[3], decoder_channels],
            out_channels=decoder_channels
        )
        
        self.decoder2 = DecoderBlock(
            in_channels_list=[encoder_channels[1], encoder_channels[2], 
                            decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        
        self.decoder1 = DecoderBlock(
            in_channels_list=[encoder_channels[1], decoder_channels, 
                            decoder_channels, decoder_channels],
            out_channels=decoder_channels
        )
        # ================== CBAM ==================
        if self.is_cbam_en3:
            self.cbam_e3 = CBAM(encoder_channels[3])  # 320
        if self.is_cbam_en4:
            self.cbam_e4 = CBAM(encoder_channels[4])  # 512
        # ================== BEM_Multilevel ==================
        if self.is_bem:
            self.bem = BEM_Multilevel(
                in_channels_list=[decoder_channels] * 4,
                out_channels=decoder_channels
            )
        
        # ================== DEEP SUPERVISION HEADS ==================
        self.out_d4 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d3 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d2 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d1 = nn.Conv2d(decoder_channels, n_classes, 1)
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB image (B, 3, H, W)
            depth: Depth map (B, 1, H, W)
        
        Returns:
            predictions: (pred_d1, pred_d2, pred_d3, pred_d4)
            boundary_pred: Boundary prediction or None
        """
        input_size = rgb.shape[2:]
        
        # ================== DUAL-STREAM ENCODER WITH BFM ==================
        features = self.dual_encoder(rgb, depth)  # [F1, F2, F3, F4] fused
        
        e1 = features[0]  # 64 channels
        e2 = features[1]  # 128 channels
        e3 = self.cbam_e3(features[2]) if self.is_cbam_en3 else features[2]  # 320
        e4 = self.cbam_e4(features[3]) if self.is_cbam_en4 else features[3]  # 512
        
        # ================== DECODER ==================
        size_d4 = (e4.shape[2], e4.shape[3])
        size_d3 = (e3.shape[2], e3.shape[3])
        size_d2 = (e2.shape[2], e2.shape[3])
        size_d1 = (e1.shape[2], e1.shape[3])
        
        d4 = self.decoder4([e1, e2, e3, e4], size_d4)
        d3 = self.decoder3([e1, e2, e3, d4], size_d3)
        d2 = self.decoder2([e1, e2, d3, d4], size_d2)
        d1 = self.decoder1([e1, d2, d3, d4], size_d1)
        
        # ================== BEM_Multilevel ==================
        if self.is_bem:
            d1_up = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
            d2_up = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)
            d3_up = F.interpolate(d3, size=input_size, mode='bilinear', align_corners=True)
            d4_up = F.interpolate(d4, size=input_size, mode='bilinear', align_corners=True)
            
            decoder_output, boundary_pred = self.bem([d1_up, d2_up, d3_up, d4_up])
        else:
            decoder_output = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
            boundary_pred = None
        
        # ================== DEEP SUPERVISION OUTPUTS ==================
        pred_d4 = F.interpolate(self.out_d4(d4), size=input_size, mode='bilinear', align_corners=True)
        pred_d3 = F.interpolate(self.out_d3(d3), size=input_size, mode='bilinear', align_corners=True)
        pred_d2 = F.interpolate(self.out_d2(d2), size=input_size, mode='bilinear', align_corners=True)
        pred_d1 = self.out_d1(decoder_output)
        
        return (pred_d1, pred_d2, pred_d3, pred_d4), boundary_pred
