import torch


def f_measure(pred, gt, beta2=0.3*0.3, eps=1e-7):
    """
    Paper-level Fβ score for camouflage/saliency
    pred, gt: (1,H,W) or (H,W) in [0,1]

    Steps:
    1) Adaptive threshold = 2 * mean(pred)
    2) Binarize
    3) Compute precision / recall
    """
    pred = pred.squeeze().float()
    gt   = gt.squeeze().float()

    # adaptive threshold
    thr = 2 * pred.mean()
    bin_pred = (pred >= thr).float()

    tp = (bin_pred * gt).sum()
    fp = (bin_pred * (1 - gt)).sum()
    fn = ((1 - bin_pred) * gt).sum()

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)

    fbeta = (1 + beta2) * prec * rec / (beta2 * prec + rec + eps)
    return fbeta
