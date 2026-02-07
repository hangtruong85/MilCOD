"""
Dual-Stream PVT for Camouflaged Object Detection
OPTIMIZED VERSION

Key Optimizations:
1. Spatial Reduction Attention (SRA) - Giảm 64x memory ở Stage 1
2. Spatial Attention Mask - Identify reliable depth regions
3. Pretrained weight adaptation cho depth stream (1-channel)
4. Kaiming initialization cho Conv layers
5. Temperature scaling và LayerNorm trong attention

Architecture:
    - RGB Stream: PVT-v2-b2 (3ch → [64, 128, 320, 512])
    - Depth Stream: PVT-v2-b0 (1ch → [32, 64, 160, 256])
    - BFM with SRA at each stage
    - Full-scale skip connections decoder
    - CBAM + BEM + Deep supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from .boundary_enhancement import BoundaryEnhancementModule, BEM_Multilevel
from .cbam import CBAM


# ============================================================================
# Pretrained Weight Loader for Depth Stream
# ============================================================================
def load_pretrained_depth_backbone(model_name='pvt_v2_b0'):
    """
    Load pretrained weights và adapt cho 1-channel depth input
    
    Strategy: Average RGB channels (3ch → 1ch) cho first conv layer
    """
    # Load RGB pretrained model
    model_rgb = timm.create_model(model_name, pretrained=True, features_only=True)
    
    # Create depth model (1-channel)
    model_depth = timm.create_model(
        model_name, 
        pretrained=False, 
        features_only=True,
        in_chans=1
    )
    
    # Get state dicts
    rgb_state = model_rgb.state_dict()
    depth_state = model_depth.state_dict()
    
    # Adapt first conv layer
    first_conv_adapted = False
    for key in rgb_state.keys():
        if 'weight' in key and len(rgb_state[key].shape) == 4:
            if rgb_state[key].shape[1] == 3:  # First conv (3 input channels)
                # Average RGB channels → 1 channel
                depth_state[key] = rgb_state[key].mean(dim=1, keepdim=True)
                print(f"  Adapted {key}: {rgb_state[key].shape} → {depth_state[key].shape}")
                first_conv_adapted = True
                break
    
    # Copy remaining weights
    for key in rgb_state.keys():
        if key in depth_state:
            if rgb_state[key].shape == depth_state[key].shape:
                depth_state[key] = rgb_state[key]
    
    # Load adapted weights
    model_depth.load_state_dict(depth_state, strict=False)
    
    if first_conv_adapted:
        print("  ✓ Pretrained weights adapted for depth stream")
    
    return model_depth


# ============================================================================
# Spatial Reduction Attention (SRA)
# ============================================================================
class SpatialReductionAttention(nn.Module):
    """
    Spatial Reduction Attention để giảm memory usage
    
    Thay vì attention full (HW × HW), giảm spatial dimension của K,V
    
    Memory savings:
    - 88×88: reduction_ratio=8 → 64× smaller
    - 44×44: reduction_ratio=4 → 16× smaller
    """
    def __init__(self, channels, reduction_ratio=1):
        super().__init__()
        
        self.reduction_ratio = reduction_ratio
        
        # Q, K, V projections
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Spatial reduction for K, V
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(
                channels, 
                channels, 
                kernel_size=reduction_ratio, 
                stride=reduction_ratio
            )
            self.norm = nn.LayerNorm(channels)
        
        self.scale = (channels // 8) ** -0.5
    
    def forward(self, q_feat, kv_feat):
        """
        Args:
            q_feat: Query features (B, C, H, W)
            kv_feat: Key/Value features (B, C, H, W)
        
        Returns:
            out: Attention output (B, C, H, W)
        """
        B, C, H, W = q_feat.shape
        
        # Query
        Q = self.query(q_feat).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        
        # Key, Value with spatial reduction
        if self.reduction_ratio > 1:
            kv_feat_reduced = self.sr(kv_feat)  # (B, C, H/r, W/r)
            kv_feat_reduced = self.norm(
                kv_feat_reduced.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
            
            Hr, Wr = kv_feat_reduced.shape[2], kv_feat_reduced.shape[3]
            K = self.key(kv_feat_reduced).view(B, -1, Hr * Wr)  # (B, C', HW/r²)
            V = self.value(kv_feat_reduced).view(B, -1, Hr * Wr)  # (B, C, HW/r²)
        else:
            K = self.key(kv_feat).view(B, -1, H * W)  # (B, C', HW)
            V = self.value(kv_feat).view(B, -1, H * W)  # (B, C, HW)
        
        # Attention with temperature scaling
        attn = torch.bmm(Q, K) * self.scale  # (B, HW, HW/r²)
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)
        
        return out


# ============================================================================
# Spatial Attention Mask
# ============================================================================
class SpatialAttentionMask(nn.Module):
    """
    Spatial Attention Mask để identify reliable depth regions
    
    Output: Spatial mask [0, 1] where:
    - High values: depth reliable
    - Low values: depth noisy/unreliable
    """
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        
        total_channels = rgb_channels + depth_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, 3, padding=1),
            nn.BatchNorm2d(total_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Kaiming initialization
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, f_rgb, f_depth):
        """Generate spatial reliability mask"""
        concat = torch.cat([f_rgb, f_depth], dim=1)
        mask = self.conv(concat)
        return mask


# ============================================================================
# FFT Frequency Filter - Lọc bỏ nền bằng tần số Fourier
# ============================================================================
class FFTFrequencyFilter(nn.Module):
    """
    FFT-based high-pass filter cho feature maps ở Stage 4 (11×11 hoặc 7×7).
    
    Ý tưởng: Trong ảnh ngụy trang, nền thường chiếm thành phần tần số thấp
    (smooth, uniform), còn vật thể ngụy trang có biên và texture tạo ra
    thành phần tần số cao. Module này:
    1. Chuyển feature map sang miền tần số (FFT 2D)
    2. Áp dụng learnable high-pass mask để lọc bỏ low-freq (nền)
    3. Chuyển ngược về spatial domain (iFFT)
    4. Gộp với feature gốc qua learnable gate
    
    Args:
        channels: Số channels của feature map
        cutoff_ratio: Tỷ lệ vùng tần số thấp bị triệt (0~1), default 0.25
    """
    def __init__(self, channels, cutoff_ratio=0.25):
        super().__init__()
        
        self.channels = channels
        self.cutoff_ratio = cutoff_ratio
        
        # Learnable threshold để điều chỉnh mức lọc cho từng channel
        # Khởi tạo = cutoff_ratio, model sẽ tự học giá trị tối ưu
        self.freq_threshold = nn.Parameter(torch.ones(1, channels, 1, 1) * cutoff_ratio)
        
        # Gate để blend giữa original và filtered features
        # Khởi tạo nhỏ (0.1) để ban đầu chủ yếu giữ original, tránh phá vỡ pretrained features
        self.gate = nn.Parameter(torch.ones(1) * 0.1)
        
        # 1x1 Conv để refine sau khi lọc FFT
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.refine[0].weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W) - typically 11×11 or 7×7 at Stage 4
        
        Returns:
            out: High-freq enhanced feature map (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1) FFT 2D trên spatial dimensions (H, W)
        # Output: complex tensor (B, C, H, W)
        x_freq = torch.fft.fft2(x, norm='ortho')
        
        # 2) Shift zero-frequency component to center
        x_freq_shifted = torch.fft.fftshift(x_freq, dim=(-2, -1))
        
        # 3) Tạo high-pass mask (suppress low frequencies ở center)
        # Learnable cutoff per channel, clamped to [0.05, 0.5]
        cutoff = torch.clamp(self.freq_threshold, 0.05, 0.5)
        
        # Tạo distance map từ center (normalized 0~1)
        cy, cx = H // 2, W // 2
        y_coords = torch.arange(H, device=x.device, dtype=x.dtype) - cy
        x_coords = torch.arange(W, device=x.device, dtype=x.dtype) - cx
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalize distance to [0, 1]
        dist = torch.sqrt(yy ** 2 + xx ** 2)
        max_dist = torch.sqrt(torch.tensor(cy ** 2 + cx ** 2, dtype=x.dtype, device=x.device))
        dist_normalized = dist / (max_dist + 1e-6)  # (H, W)
        
        # High-pass mask: 1 ở high-freq, 0 ở low-freq
        # Smooth transition bằng sigmoid thay vì hard cutoff
        # cutoff shape: (1, C, 1, 1), dist_normalized shape: (H, W)
        steepness = 10.0  # Điều chỉnh độ dốc của transition
        hp_mask = torch.sigmoid(steepness * (dist_normalized.unsqueeze(0).unsqueeze(0) - cutoff))
        # hp_mask shape: (1, C, H, W) - broadcast cho batch
        
        # 4) Apply mask trong frequency domain
        x_freq_filtered = x_freq_shifted * hp_mask
        
        # 5) Inverse shift + iFFT để trở về spatial domain
        x_freq_unshifted = torch.fft.ifftshift(x_freq_filtered, dim=(-2, -1))
        x_filtered = torch.fft.ifft2(x_freq_unshifted, norm='ortho').real
        
        # 6) Refine filtered features
        x_filtered = self.refine(x_filtered)
        
        # 7) Gated residual: blend original và filtered
        # gate clamp [0, 1] để ổn định
        g = torch.clamp(self.gate, 0.0, 1.0)
        out = x + g * x_filtered
        
        return out


# ============================================================================
# Bi-directional Fusion Module (BFM) - OPTIMIZED
# ============================================================================
class BiDirectionalFusionModule(nn.Module):
    """
    BFM - OPTIMIZED VERSION
    
    Improvements:
    1. Spatial Reduction Attention (SRA) cho memory efficiency
    2. Spatial Attention Mask cho reliable region detection
    3. Temperature scaling + LayerNorm
    4. Kaiming initialization
    
    Args:
        rgb_channels: RGB feature channels
        depth_channels: Depth feature channels
        out_channels: Output channels
        reduction_ratio: Spatial reduction ratio for SRA
        use_spatial_mask: Whether to use spatial attention mask
    """
    def __init__(self, rgb_channels, depth_channels, out_channels=None, 
                 reduction_ratio=1, use_spatial_mask=True):
        super().__init__()
        
        if out_channels is None:
            out_channels = rgb_channels
        
        self.use_spatial_mask = use_spatial_mask
        
        # ========== Channel Alignment ==========
        if depth_channels != rgb_channels:
            self.align_depth = nn.Sequential(
                nn.Conv2d(depth_channels, rgb_channels, kernel_size=1),
                nn.BatchNorm2d(rgb_channels),
                nn.ReLU(inplace=True)
            )
            nn.init.kaiming_normal_(
                self.align_depth[0].weight, 
                mode='fan_out', 
                nonlinearity='relu'
            )
        else:
            self.align_depth = nn.Identity()
        
        # ========== Spatial Attention Mask ==========
        if use_spatial_mask:
            self.spatial_mask = SpatialAttentionMask(rgb_channels, rgb_channels)
        
        # ========== D2R Attention (with SRA) ==========
        self.d2r_attn = SpatialReductionAttention(rgb_channels, reduction_ratio)
        self.d2r_gamma = nn.Parameter(torch.ones(1) * 0.1)
        self.d2r_norm = nn.LayerNorm([rgb_channels])
        
        # ========== R2D Attention (with SRA) ==========
        self.r2d_attn = SpatialReductionAttention(rgb_channels, reduction_ratio)
        self.r2d_gamma = nn.Parameter(torch.ones(1) * 0.1)
        self.r2d_norm = nn.LayerNorm([rgb_channels])
        
        # ========== Fusion ==========
        fusion_in_channels = rgb_channels * 2
        if use_spatial_mask:
            fusion_in_channels += 1  # Add mask channel
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        nn.init.kaiming_normal_(
            self.fusion_conv[0].weight, 
            mode='fan_out', 
            nonlinearity='relu'
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
        
        # Align depth
        f_depth_aligned = self.align_depth(f_depth)
        
        # Spatial alignment
        if f_depth_aligned.shape[2:] != f_rgb.shape[2:]:
            f_depth_aligned = F.interpolate(
                f_depth_aligned, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=True
            )
        
        # ========== Spatial Attention Mask ==========
        if self.use_spatial_mask:
            spatial_mask = self.spatial_mask(f_rgb, f_depth_aligned)
            f_depth_masked = f_depth_aligned * spatial_mask
        else:
            f_depth_masked = f_depth_aligned
            spatial_mask = None
        
        # ========== D2R: Depth-to-RGB ==========
        out_d2r = self.d2r_attn(f_rgb, f_depth_masked)
        out_d2r = self.d2r_norm(out_d2r.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_rgb_enhanced = f_rgb + torch.clamp(self.d2r_gamma, 0, 1.0) * out_d2r
        
        # ========== R2D: RGB-to-Depth ==========
        out_r2d = self.r2d_attn(f_depth_masked, f_rgb)
        out_r2d = self.r2d_norm(out_r2d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_depth_enhanced = f_depth_aligned + torch.clamp(self.r2d_gamma, 0, 1.0) * out_r2d
        
        # ========== Fusion ==========
        if self.use_spatial_mask:
            f_fusion = torch.cat([f_rgb_enhanced, f_depth_enhanced, spatial_mask], dim=1)
        else:
            f_fusion = torch.cat([f_rgb_enhanced, f_depth_enhanced], dim=1)
        
        f_fusion = self.fusion_conv(f_fusion)
        
        return f_fusion


# ============================================================================
# Decoder Block
# ============================================================================
class DecoderBlock(nn.Module):
    """
    Decoder Block with full-scale skip connections
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        cat_channels = out_channels * len(in_channels_list)
        
        self.conv_branches = nn.ModuleList()
        for in_ch in in_channels_list:
            conv = nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            # Kaiming initialization
            nn.init.kaiming_normal_(conv[0].weight, mode='fan_out', nonlinearity='relu')
            self.conv_branches.append(conv)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(cat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.fusion[0].weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, features, target_size):
        processed = []
        
        for feat, conv in zip(features, self.conv_branches):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            feat = conv(feat)
            processed.append(feat)
        
        cat_feat = torch.cat(processed, dim=1)
        out = self.fusion(cat_feat)
        
        return out


# ============================================================================
# Dual-Stream Encoder
# ============================================================================
class DualStreamEncoder(nn.Module):
    """
    Dual-Stream Encoder - OPTIMIZED
    
    Features:
    - Pretrained weight adaptation for depth stream
    - SRA with adaptive reduction ratios:
      * Stage 1 (88×88): reduction_ratio=8 → 64× memory save
      * Stage 2 (44×44): reduction_ratio=4 → 16× memory save
      * Stage 3 (22×22): reduction_ratio=2 → 4× memory save
      * Stage 4 (11×11): reduction_ratio=1 (no reduction)
    - Spatial attention mask at each stage
    """
    def __init__(self, pretrained=True, use_spatial_mask=True, is_fft=False):
        super().__init__()
        
        self.is_fft = is_fft
        # ========== RGB Stream (PVT-v2-b2) ==========
        self.rgb_backbone = timm.create_model(
            'pvt_v2_b2', 
            pretrained=pretrained, 
            features_only=True
        )
        self.rgb_channels = [64, 128, 320, 512]
        
        # ========== Depth Stream (PVT-v2-b0) with Adapted Weights ==========
        if pretrained:
            print("Loading pretrained weights for depth stream...")
            self.depth_backbone = load_pretrained_depth_backbone('pvt_v2_b0')
        else:
            self.depth_backbone = timm.create_model(
                'pvt_v2_b0', 
                pretrained=False, 
                features_only=True,
                in_chans=1
            )
        self.depth_channels = [32, 64, 160, 256]
        
        # ========== BFM with Adaptive SRA ==========
        self.bfm1 = BiDirectionalFusionModule(
            rgb_channels=self.rgb_channels[0],
            depth_channels=self.depth_channels[0],
            out_channels=self.rgb_channels[0],
            reduction_ratio=8,  # 88×88 → 11×11 (64× memory save)
            use_spatial_mask=use_spatial_mask
        )
        
        self.bfm2 = BiDirectionalFusionModule(
            rgb_channels=self.rgb_channels[1],
            depth_channels=self.depth_channels[1],
            out_channels=self.rgb_channels[1],
            reduction_ratio=4,  # 44×44 → 11×11 (16× memory save)
            use_spatial_mask=use_spatial_mask
        )
        
        self.bfm3 = BiDirectionalFusionModule(
            rgb_channels=self.rgb_channels[2],
            depth_channels=self.depth_channels[2],
            out_channels=self.rgb_channels[2],
            reduction_ratio=2,  # 22×22 → 11×11 (4× memory save)
            use_spatial_mask=use_spatial_mask
        )
        
        self.bfm4 = BiDirectionalFusionModule(
            rgb_channels=self.rgb_channels[3],
            depth_channels=self.depth_channels[3],
            out_channels=self.rgb_channels[3],
            reduction_ratio=1,  # 11×11 (already small, no reduction)
            use_spatial_mask=use_spatial_mask
        )
        
        # ========== FFT Frequency Filter (Stage 4 only) ==========
        if self.is_fft:
            self.fft_filter = FFTFrequencyFilter(
                channels=self.rgb_channels[3],  # 512
                cutoff_ratio=0.25
            )
            print("  ✓ FFT Frequency Filter enabled at Stage 4")
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB image (B, 3, H, W)
            depth: Depth map (B, 1, H, W)
        
        Returns:
            fused_features: List [F1, F2, F3, F4]
        """
        # Extract features
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        
        # Bi-directional fusion with SRA
        f1_fusion = self.bfm1(rgb_features[0], depth_features[0])
        f2_fusion = self.bfm2(rgb_features[1], depth_features[1])
        f3_fusion = self.bfm3(rgb_features[2], depth_features[2])
        f4_fusion = self.bfm4(rgb_features[3], depth_features[3])
        
        # Apply FFT high-pass filter at Stage 4 (before decoder)
        if self.is_fft:
            f4_fusion = self.fft_filter(f4_fusion)
        
        return [f1_fusion, f2_fusion, f3_fusion, f4_fusion]


# ============================================================================
# Complete Model
# ============================================================================
class DualStream_PVT_COD(nn.Module):
    """
    Dual-Stream PVT-COD - COMPLETE OPTIMIZED VERSION
    
    Full architecture:
    - Dual-Stream Encoder với SRA và Spatial Mask
    - Full-scale skip decoder
    - CBAM attention
    - BEM boundary enhancement
    - Deep supervision
    
    Optimizations:
    1. Memory: 64× reduction ở Stage 1 (SRA)
    2. Depth reliability: Spatial attention mask
    3. Pretrained: Adapted weights cho depth stream
    4. Initialization: Kaiming normal cho Conv layers
    5. Stability: LayerNorm + temperature scaling
    6. FFT Filter: High-pass frequency filter ở Stage 4 để lọc nền
    
    Args:
        n_classes: Number of output classes (default: 1)
        is_bem: Use BEM (default: True)
        is_cbam_en3: Use CBAM on stage 3 (default: True)
        is_cbam_en4: Use CBAM on stage 4 (default: True)
        pretrained: Use pretrained backbones (default: True)
        use_spatial_mask: Use spatial attention mask (default: True)
        is_fft: Use FFT frequency filter at Stage 4 (default: False)
    """
    def __init__(self, n_classes=1, is_bem=True, is_cbam_en3=True, 
                 is_cbam_en4=True, pretrained=True, use_spatial_mask=True,
                 is_fft=False):
        super().__init__()
        
        # ================== DUAL-STREAM ENCODER ==================
        self.dual_encoder = DualStreamEncoder(
            pretrained=pretrained,
            use_spatial_mask=use_spatial_mask,
            is_fft=is_fft
        )
        
        encoder_channels = [64, 128, 320, 512]
        
        # ================== FLAGS ==================
        self.is_bem = is_bem
        self.is_cbam_en3 = is_cbam_en3
        self.is_cbam_en4 = is_cbam_en4
        
        # ================== CBAM ==================
        if self.is_cbam_en3:
            self.cbam_e3 = CBAM(encoder_channels[2])
        if self.is_cbam_en4:
            self.cbam_e4 = CBAM(encoder_channels[3])
        
        # ================== DECODER ==================
        decoder_channels = 64
        
        self.decoder4 = DecoderBlock(encoder_channels, decoder_channels)
        self.decoder3 = DecoderBlock(
            [encoder_channels[0], encoder_channels[1], encoder_channels[2], decoder_channels],
            decoder_channels
        )
        self.decoder2 = DecoderBlock(
            [encoder_channels[0], encoder_channels[1], decoder_channels, decoder_channels],
            decoder_channels
        )
        self.decoder1 = DecoderBlock(
            [encoder_channels[0], decoder_channels, decoder_channels, decoder_channels],
            decoder_channels
        )
        
        # ================== BEM ==================
        if self.is_bem:
            self.bem = BEM_Multilevel(
                in_channels_list=[decoder_channels] * 4,
                out_channels=decoder_channels
            )
        
        # ================== OUTPUT HEADS ==================
        self.out_d4 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d3 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d2 = nn.Conv2d(decoder_channels, n_classes, 1)
        self.out_d1 = nn.Conv2d(decoder_channels, n_classes, 1)
        
        # Initialize output biases to small positive value
        for head in [self.out_d1, self.out_d2, self.out_d3, self.out_d4]:
            nn.init.constant_(head.bias, 0.01)
    
    def forward(self, rgb, depth):
        """
        Forward pass
        
        Args:
            rgb: RGB image (B, 3, H, W)
            depth: Depth map (B, 1, H, W)
        
        Returns:
            predictions: Tuple (pred_d1, pred_d2, pred_d3, pred_d4)
            boundary_pred: Boundary prediction or None
        """
        input_size = rgb.shape[2:]
        
        # ================== ENCODER ==================
        features = self.dual_encoder(rgb, depth)
        
        e1 = features[0]
        e2 = features[1]
        e3 = self.cbam_e3(features[2]) if self.is_cbam_en3 else features[2]
        e4 = self.cbam_e4(features[3]) if self.is_cbam_en4 else features[3]
        
        # ================== DECODER ==================
        size_d4 = (e4.shape[2], e4.shape[3])
        size_d3 = (e3.shape[2], e3.shape[3])
        size_d2 = (e2.shape[2], e2.shape[3])
        size_d1 = (e1.shape[2], e1.shape[3])
        
        d4 = self.decoder4([e1, e2, e3, e4], size_d4)
        d3 = self.decoder3([e1, e2, e3, d4], size_d3)
        d2 = self.decoder2([e1, e2, d3, d4], size_d2)
        d1 = self.decoder1([e1, d2, d3, d4], size_d1)
        
        # ================== BEM ==================
        if self.is_bem:
            d1_up = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
            d2_up = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)
            d3_up = F.interpolate(d3, size=input_size, mode='bilinear', align_corners=True)
            d4_up = F.interpolate(d4, size=input_size, mode='bilinear', align_corners=True)
            
            decoder_output, boundary_pred = self.bem([d1_up, d2_up, d3_up, d4_up])
        else:
            decoder_output = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
            boundary_pred = None
        
        # ================== OUTPUTS ==================
        pred_d4 = F.interpolate(self.out_d4(d4), size=input_size, mode='bilinear', align_corners=True)
        pred_d3 = F.interpolate(self.out_d3(d3), size=input_size, mode='bilinear', align_corners=True)
        pred_d2 = F.interpolate(self.out_d2(d2), size=input_size, mode='bilinear', align_corners=True)
        pred_d1 = self.out_d1(decoder_output)
        
        return (pred_d1, pred_d2, pred_d3, pred_d4), boundary_pred