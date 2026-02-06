import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


def gaussian_kernel(channel, kernel_size=7, sigma=5):
    """Create 2D Gaussian kernel"""
    grid = torch.arange(kernel_size).float()
    mean = (kernel_size - 1) / 2
    kernel = torch.exp(-(grid - mean) ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    g2d = torch.outer(kernel, kernel)
    g2d = g2d.unsqueeze(0).unsqueeze(0)
    return g2d.repeat(channel, 1, 1, 1)


def bwdist_torch(mask):
    """
    Compute distance transform using scipy (for accuracy with MATLAB)
    mask: binary mask (1 = foreground, 0 = background)
    Returns: distance map
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)
    
    # Distance from background (0) to foreground (1)
    dist = distance_transform_edt(mask_np)
    return torch.from_numpy(dist).float().to(mask.device)


class WeightedFmeasureTorch:
    """
    Weighted F-measure from CVPR 2014 - PyTorch implementation
    
    Exact match with MATLAB reference implementation
    """
    def __init__(self, beta=1.0):
        self.beta = beta
        self.weighted_fms = []
    
    def step(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred, gt: shape (B, H, W) or (H, W), values in [0, 1]
        """
        pred = pred.squeeze().float()
        gt = gt.squeeze().float()
        
        # Nếu toàn bộ gt = 0 (background), wfm = 0
        if torch.all(gt == 0):
            wfm = 0.0
        else:
            wfm = self.cal_wfm(pred, gt)
        
        self.weighted_fms.append(float(wfm))
    
    def cal_wfm(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted F-measure
        
        pred, gt: (H, W), values in [0, 1]
        """
        device = pred.device
        
        # Step 1: Compute distance transform from background
        # [Dst, Idxt] = bwdist(dGT);  where dGT = (gt == 0)
        gt_bg = (gt == 0).float()  # Background mask
        Dst = bwdist_torch(gt_bg)
        
        # Step 2: Compute error map
        # E = abs(FG - dGT);
        E = torch.abs(pred - gt)
        
        # Step 3: Error propagation from boundary
        # Et = E; Et(gt==0) = Et(indices of nearest foreground)
        Et = E.clone()
        
        if torch.any(gt == 0):
            gt_fg = (gt == 1).float()
            # Distance transform of foreground
            Dst_fg = bwdist_torch(gt_fg)
            
            kernel = gaussian_kernel(1, kernel_size=7, sigma=5).to(device)
            E_dilated = F.conv2d(E.unsqueeze(0).unsqueeze(0), kernel, padding=3)[0, 0]
            
            Et = torch.where(gt == 0, E_dilated, E)
        
        # Step 4: Gaussian blur
        # K = fspecial('gaussian', 7, 5);
        # EA = imfilter(Et, K);
        kernel = gaussian_kernel(1, kernel_size=7, sigma=5).to(device)
        EA = F.conv2d(Et.unsqueeze(0).unsqueeze(0), kernel, padding=3)[0, 0]
        
        # Step 5: Min error strategy
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = torch.where(
            (gt == 1) & (EA < E),
            EA,
            E
        )
        
        # Step 6: Compute importance weight B
        # B = ones_like(gt);
        # B(gt==0) = 2 - exp(log(0.5) / 5 * Dst);
        B = torch.ones_like(gt)
        B[gt == 0] = 2 - torch.exp(torch.tensor(np.log(0.5) / 5) * Dst[gt == 0])
        
        # Step 7: Weighted error
        # Ew = MIN_E_EA * B
        Ew = MIN_E_EA * B
        
        # Step 8: Calculate metrics
        # TPw = sum(gt) - sum(Ew[gt==1])
        # FPw = sum(Ew[gt==0])
        TPw = torch.sum(gt) - torch.sum(Ew[gt == 1])
        FPw = torch.sum(Ew[gt == 0])
        
        # Step 9: Recall and Precision
        # R = 1 - mean(Ew[gt==1])
        # P = TPw / (TPw + FPw + eps)
        gt_fg_count = torch.sum(gt == 1)
        if gt_fg_count > 0:
            R = 1 - torch.mean(Ew[gt == 1])
        else:
            R = torch.tensor(0.0, device=device)
        
        eps = 1e-7
        P = TPw / (TPw + FPw + eps)
        
        # Step 10: F-measure
        # Q = (1 + beta^2) * (R * P) / (R + beta * P + eps)
        Q = (1 + self.beta ** 2) * R * P / (R + self.beta ** 2 * P + eps)
        
        return Q
    
    def get_results(self) -> dict:
        """Get average weighted F-measure"""
        weighted_fm = np.mean(np.array(self.weighted_fms))
        return {'wfm': weighted_fm}


def fw_measure(pred: torch.Tensor, gt: torch.Tensor, beta=1.0) -> torch.Tensor:
    """
    Single image weighted F-measure calculation
    
    pred, gt: (H, W) or (1, H, W), values in [0, 1]
    beta: F-measure beta parameter (default 1.0 for F1-measure)
    
    Returns: scalar tensor
    """
    calculator = WeightedFmeasureTorch(beta=beta)
    calculator.step(pred, gt)
    results = calculator.get_results()
    return torch.tensor(results['wfm'])


def fw_measure_old(pred, gt, beta2=0.3*0.3):
    """
    pred, gt ∈ [0,1], shape (1,H,W)
    Weighted F-measure from CVPR2014

    Steps:
    1) absolute error map
    2) spatial weighting (Gaussian)
    3) weighted precision / recall
    """

    pred = pred.squeeze().float()
    gt   = gt.squeeze().float()

    # step1: error map
    E = torch.abs(pred - gt)

    # step2: Gaussian blur
    g = gaussian_kernel(1, kernel_size=7, sigma=5).to(pred.device)
    E_blur = F.conv2d(E.unsqueeze(0).unsqueeze(0), g, padding=3)[0,0]

    # step3: Weighting
    TP = (gt * (1 - E_blur)).sum()
    FP = ((1 - gt) * (1 - E_blur)).sum()

    FN = (gt * E_blur).sum()

    P_w = TP / (TP + FP + 1e-7)
    R_w = TP / (TP + FN + 1e-7)

    F_w = (1 + beta2) * P_w * R_w / (beta2 * P_w + R_w + 1e-7)
    return F_w
