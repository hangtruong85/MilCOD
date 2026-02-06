"""
- mae_metric: Mean Absolute Error metric
"""
import torch

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