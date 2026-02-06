import torch


def e_measure(pred, gt):
    pred = pred.squeeze().float()
    gt = gt.squeeze().float()

    pred_mean = pred.mean()
    gt_mean = gt.mean()

    pred_hat = pred - pred_mean
    gt_hat = gt - gt_mean

    align = (2 * pred_hat * gt_hat + 1e-8) / (
        pred_hat**2 + gt_hat**2 + 1e-8
    )
    enhanced = ((align + 1)**2) / 4
    return enhanced.mean()
