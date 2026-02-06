"""
Loss functions for camouflaged object detection
- FocalLoss: Focal Loss from ICCV 2017
- SegmentationLoss: Combined loss (BCE + Dice + IoU + Focal + Boundary)
- mae_metric: Mean Absolute Error metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weight for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               gamma=0 -> equivalent to BCE
               gamma>0 -> down-weights easy examples
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
        # p_t = p if y=1, else 1-p
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        # Calculate alpha_t
        # alpha_t = alpha if y=1, else 1-alpha
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


class SegmentationLoss(nn.Module):
    """
    Combined loss for camouflaged object detection
    Includes: BCE + Dice + IoU + Focal + Boundary loss
    
    Args:
        lambda_bce: Weight for BCE loss (default: 1.0)
        lambda_dice: Weight for Dice loss (default: 1.0)
        lambda_iou: Weight for IoU loss (default: 0.5)
        lambda_focal: Weight for Focal loss (default: 1.0)
        lambda_boundary: Weight for boundary loss (default: 0.3)
        focal_alpha: Alpha parameter for Focal loss (default: 0.25)
        focal_gamma: Gamma parameter for Focal loss (default: 2.0)
    """
    def __init__(self, lambda_bce=1.0, lambda_dice=1.0, lambda_iou=0.5, 
                 lambda_focal=1.0, lambda_boundary=0.3,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_iou = lambda_iou
        self.lambda_focal = lambda_focal
        self.lambda_boundary = lambda_boundary
        
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def dice_loss(self, pred, target):
        """
        Dice loss for binary segmentation
        
        Dice = 2 * (intersection) / (union)
        Dice Loss = 1 - Dice
        """
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-7
        
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def iou_loss(self, pred, target):
        """
        IoU (Intersection over Union) loss
        
        IoU = intersection / union
        IoU Loss = 1 - IoU
        """
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-7
        
        intersection = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou
    
    def boundary_loss(self, boundary_pred, boundary_target):
        """
        Boundary loss (L1 + BCE)
        
        Args:
            boundary_pred: predicted boundary logits (B, 1, H, W)
            boundary_target: ground truth boundary map (B, 1, H, W)
        """
        # L1 loss on probability space
        pred_prob = torch.sigmoid(boundary_pred)
        l1_loss = F.l1_loss(pred_prob, boundary_target)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(boundary_pred, boundary_target)
        
        return l1_loss + bce_loss
    
    def forward(self, pred, target, boundary_pred=None, boundary_target=None):
        """
        Forward pass for loss computation
        
        Args:
            pred: predicted mask logits (B, 1, H, W)
            target: ground truth mask (B, 1, H, W)
            boundary_pred: predicted boundary logits (B, 1, H, W) - optional
            boundary_target: ground truth boundary (B, 1, H, W) - optional
        
        Returns:
            total_loss: weighted sum of all losses
        """
        # Main segmentation losses
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice_loss(pred, target)
        loss_iou = self.iou_loss(pred, target)
        loss_focal = self.focal(pred, target)
        
        total_loss = (self.lambda_bce * loss_bce + 
                     self.lambda_dice * loss_dice + 
                     self.lambda_iou * loss_iou +
                     self.lambda_focal * loss_focal)
        
        # Add boundary loss if provided
        if boundary_pred is not None and boundary_target is not None:
            loss_boundary = self.boundary_loss(boundary_pred, boundary_target)
            total_loss += self.lambda_boundary * loss_boundary
        
        return total_loss


def mae_metric(pred, target):
    """
    Mean Absolute Error metric
    
    Args:
        pred: predicted logits (B, 1, H, W)
        target: ground truth (B, 1, H, W)
    
    Returns:
        mae: mean absolute error value
    """
    pred_prob = torch.sigmoid(pred)
    return torch.mean(torch.abs(pred_prob - target))