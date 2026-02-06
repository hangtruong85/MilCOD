"""
Deep Supervision Implementation for Segmentation
- Multiple outputs from decoder levels (d1, d2, d3, d4)
- Individual loss for each level using Weighted BCE and Weighted IoU
- Boundary supervision loss (Dice-based)
- Combined total loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisionLoss(nn.Module):
    def __init__(self, lambda_wbce=0.5, lambda_wiou=0.5):
        super().__init__()
        self.lambda_wbce = lambda_wbce
        self.lambda_wiou = lambda_wiou
    
    def compute_boundary_weight(self, mask):
        """
        Tạo weight map - cao ở vùng biên, thấp ở vùng phẳng
        
        Returns:
            weit: weight map, giá trị từ 1 (vùng phẳng) đến 6 (vùng biên)
        """
        # Blur mask để tìm vùng biên
        blurred = F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)
        
        # |blur - mask| cao ở biên, thấp ở vùng phẳng
        boundary = torch.abs(blurred - mask)
        
        # Weight: 1 (vùng phẳng) → 6 (vùng biên)
        weit = 1 + 5 * boundary
        
        return weit
    
    def weighted_bce_loss(self, pred, target, weit):
        """
        Weighted BCE - pixel ở biên có loss cao hơn
        """
        # BCE per pixel
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Weighted sum, normalized by total weight
        wbce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        
        return wbce.mean()
    
    def weighted_iou_loss(self, pred, target, weit):
        """
        Weighted IoU - pixel ở biên đóng góp nhiều hơn
        """
        pred_prob = torch.sigmoid(pred)
        
        # Weighted intersection và union
        inter = ((pred_prob * target) * weit).sum(dim=(2, 3))
        union = ((pred_prob + target) * weit).sum(dim=(2, 3))
        
        # IoU loss
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        return wiou.mean()
    
    def forward(self, pred, target):
        """
        Args:
            pred: logits (B, 1, H, W)
            target: ground truth (B, 1, H, W)
        """
        # Tính weight map dựa trên boundary
        weit = self.compute_boundary_weight(target)
        
        # Weighted losses
        loss_bce = self.weighted_bce_loss(pred, target, weit)
        loss_iou = self.weighted_iou_loss(pred, target, weit)
        
        total_loss = self.lambda_wbce * loss_bce + self.lambda_wiou * loss_iou
        
        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss combining multiple decoder levels
    
    Formula:
        L_total = L_d4 + L_d3 + L_d2 + L_d1 + 4×L_boundary
    
    Where:
        - L_d1, L_d2, L_d3, L_d4: Supervision loss at each decoder level
        - L_boundary: Boundary Dice loss
    
    Args:
        lambda_wbce: Weight for BCE in supervision loss (default: 0.25)
        lambda_wiou: Weight for IoU in supervision loss (default: 0.4)
        lambda_boundary: Weight multiplier for boundary loss (default: 4.0)
        focal_alpha: Alpha parameter for Focal loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal loss (default: 2.0)
    """
    def __init__(self, lambda_wbce=0.25, lambda_wiou=0.4,
                 lambda_boundary=4.0):
        super().__init__()
        
        self.lambda_boundary = lambda_boundary
        
        # Supervision loss for each decoder level
        self.supervision_loss = SupervisionLoss(
            lambda_wbce=lambda_wbce,
            lambda_wiou=lambda_wiou,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
    
    def boundary_dice_loss(self, boundary_pred, boundary_target):
        """
        Boundary loss using Dice
        
        Args:
            boundary_pred: predicted boundary logits (B, 1, H, W)
            boundary_target: ground truth boundary (B, 1, H, W)
        
        Returns:
            dice_loss: Dice loss for boundaries
        """
        pred_prob = torch.sigmoid(boundary_pred)
        smooth = 1e-7
        
        intersection = (pred_prob * boundary_target).sum()
        union = pred_prob.sum() + boundary_target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def forward(self, pred_d1, pred_d2, pred_d3, pred_d4, 
                target, boundary_pred=None, boundary_target=None):
        """
        Forward pass for deep supervision loss
        
        Args:
            pred_d1, pred_d2, pred_d3, pred_d4: predictions from decoder levels
                                                (B, 1, H, W) or (B, 1, H_i, W_i)
            target: ground truth mask (B, 1, H, W)
            boundary_pred: predicted boundary logits (optional, B, 1, H, W)
            boundary_target: ground truth boundary (optional, B, 1, H, W)
        
        Returns:
            total_loss: Combined deep supervision loss
        """
        # Resize predictions to match target size if needed
        target_size = target.shape[2:]
        
        if pred_d4.shape[2:] != target_size:
            pred_d4 = F.interpolate(pred_d4, size=target_size, 
                                   mode='bilinear', align_corners=True)
        if pred_d3.shape[2:] != target_size:
            pred_d3 = F.interpolate(pred_d3, size=target_size, 
                                   mode='bilinear', align_corners=True)
        if pred_d2.shape[2:] != target_size:
            pred_d2 = F.interpolate(pred_d2, size=target_size, 
                                   mode='bilinear', align_corners=True)
        # d1 should already be at target size
        
        # Calculate supervision loss for each decoder level
        loss_d4 = self.supervision_loss(pred_d4, target)
        loss_d3 = self.supervision_loss(pred_d3, target)
        loss_d2 = self.supervision_loss(pred_d2, target)
        loss_d1 = self.supervision_loss(pred_d1, target)
        
        # Total supervision loss
        total_supervision_loss = loss_d4 + loss_d3 + loss_d2 + loss_d1
        
        # Add boundary loss if provided
        total_loss = total_supervision_loss
        if boundary_pred is not None and boundary_target is not None:
            # Ensure boundary_target is in [0, 1]
            boundary_target = torch.clamp(boundary_target, 0, 1)
            
            # Resize if needed
            if boundary_pred.shape[2:] != target_size:
                boundary_pred = F.interpolate(boundary_pred, size=target_size,
                                             mode='bilinear', align_corners=True)
            
            loss_boundary = self.boundary_dice_loss(boundary_pred, boundary_target)
            total_loss += self.lambda_boundary * loss_boundary
        
        return total_loss