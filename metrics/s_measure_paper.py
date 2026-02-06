import torch
import torch.nn.functional as F


def _object(pred, gt):
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)

    u = gt.mean()
    if u == 0:
        return 1 - pred.mean()
    if u == 1:
        return pred.mean()

    o_fg = _ssim(pred, gt)
    return o_fg


def _ssim(pred, gt):
    pred_mean = pred.mean()
    gt_mean = gt.mean()

    pred_var = pred.var()
    gt_var = gt.var()

    cov = ((pred - pred_mean) * (gt - gt_mean)).mean()
    return (2 * pred_mean * gt_mean + 0.001) * (2 * cov + 0.001) / (
        (pred_mean ** 2 + gt_mean ** 2 + 0.001) *
        (pred_var + gt_var + 0.001)
    )


def _divide(pred, gt):
    H, W = pred.shape
    cx = int(W / 2)
    cy = int(H / 2)

    regions = [
        pred[:cy, :cx], pred[:cy, cx:], pred[cy:, :cx], pred[cy:, cx:]
    ]
    regions_gt = [
        gt[:cy, :cx], gt[:cy, cx:], gt[cy:, :cx], gt[cy:, cx:]
    ]

    weights = [r_gt.numel() / gt.numel() for r_gt in regions_gt]
    return regions, regions_gt, weights


def _region(pred, gt):
    regions, regions_gt, weights = _divide(pred, gt)
    s = 0
    for r_pred, r_gt, w in zip(regions, regions_gt, weights):
        s += w * _ssim(r_pred, r_gt)
    return s


def s_measure(pred, gt):
    """
    pred, gt ∈ [0,1], single channel
    """
    pred = pred.squeeze().float()
    gt = gt.squeeze().float()

    if gt.sum() == 0:
        return 1 - pred.mean()
    if gt.sum() == gt.numel():
        return pred.mean()

    s_object = _object(pred, gt)
    s_region = _region(pred, gt)
    return 0.5 * s_object + 0.5 * s_region
