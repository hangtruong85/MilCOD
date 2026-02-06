"""
Deep Supervision Implementation for Segmentation
- Individual loss for each level
- Boundary supervision loss (Dice-based)
- Combined total loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisionLoss(nn.Module):
    """
    Supervision Loss combining BCE, Dice, and IoU
    Used for deep supervision at each decoder level
    
    Formula:
        L_supervision = λ_bce×L_bce + λ_focal×L_focal + λ_iou×L_iou
    
    Args:
        lambda_bce: Weight for BCE loss (default: 0.25)
        lambda_focal: Weight for Focal loss (default: 0.35)
        lambda_iou: Weight for IoU loss (default: 0.4)
        focal_alpha: Alpha parameter for Focal loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal loss (default: 2.0)
    """
    def __init__(self, lambda_bce=0.25, lambda_focal=0.35, lambda_iou=0.4,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_focal = lambda_focal
        self.lambda_iou = lambda_iou
        
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def dice_loss(self, pred, target):
        """Dice loss"""
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-7
        
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def iou_loss(self, pred, target):
        """IoU loss"""
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-7
        
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou
    
    def forward(self, pred, target):
        """
        Forward pass
        
        Args:
            pred: predicted logits (B, 1, H, W)
            target: ground truth (B, 1, H, W)
        
        Returns:
            total_loss: weighted sum of losses
        """
        loss_bce = self.bce(pred, target)
        loss_focal = self.focal(pred, target)
        loss_iou = self.iou_loss(pred, target)
        
        total_loss = (self.lambda_bce * loss_bce + 
                     self.lambda_focal * loss_focal + 
                     self.lambda_iou * loss_iou)
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Args:
        alpha: Weight for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth (B, 1, H, W), values in [0, 1]
        """
        # Get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate p_t
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        # Calculate alpha_t
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Calculate focal weight
        focal_weight = alpha_t * torch.pow((1 - p_t), self.gamma)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss combining multiple decoder levels
    
    Formula:
        L_total = L_d4 + L_d3 + L_d2 + L_d1 + 4×L_boundary
    
    Where:
        - L_d1, L_d2, L_d3, L_d4: Supervision loss at each decoder level
        - L_boundary: Boundary Dice loss
    
    Args:
        lambda_bce: Weight for BCE in supervision loss (default: 0.25)
        lambda_focal: Weight for Focal in supervision loss (default: 0.35)
        lambda_iou: Weight for IoU in supervision loss (default: 0.4)
        lambda_boundary: Weight multiplier for boundary loss (default: 4.0)
        focal_alpha: Alpha parameter for Focal loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal loss (default: 2.0)
    """
    def __init__(self, lambda_bce=0.25, lambda_focal=0.35, lambda_iou=0.4,
                 lambda_boundary=4.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        
        self.lambda_boundary = lambda_boundary
        
        # Supervision loss for each decoder level
        self.supervision_loss = SupervisionLoss(
            lambda_bce=lambda_bce,
            lambda_focal=lambda_focal,
            lambda_iou=lambda_iou,
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
