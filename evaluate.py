"""
Evaluation script for DualStream_PVT_COD
Simplified version - only supports dual-stream RGB+Depth model
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from datasets.mhcd_dataset import MHCDDatasetWithDepth
from models.dual_stream_pvt import DualStream_PVT_COD

from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.fweighted_measure import fw_measure
from metrics.mae_metric import mae_metric
from utils.logger import setup_logger


# =========================================================
# Configuration
# =========================================================
class EvalConfig:
    """Evaluation configuration"""
    def __init__(self):
        # Model configuration (must match training config)
        self.n_classes = 1
        self.is_bem = True
        self.is_cbam_en3 = True
        self.is_cbam_en4 = True
        self.pretrained = False  # Not needed for evaluation
        
        # Checkpoint path
        self.ckpt_path = "logs/DualStream_PVT_COD_20260207_081637/best_s_measure.pth"
        
        # Dataset
        self.root = "../MHCD_seg"
        self.split = "test"  # or "test" if available
        self.img_size = 352
        self.batch_size = 12
        self.use_depth = True  # Load depth maps
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = f"eval_results/DualStream_PVT_COD_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)


# =========================================================
# Checkpoint Loading
# =========================================================
def load_checkpoint(model, ckpt_path, device, logger):
    """
    Load checkpoint - handles different formats
    
    Args:
        model: DualStream_PVT_COD
        ckpt_path: Path to checkpoint file
        device: cuda or cpu
        logger: Logger
    
    Returns:
        model: Model with loaded weights
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Case 1: Full checkpoint dict with 'model' key
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "unknown")
        best_metric = ckpt.get("best_s_measure", "unknown")
        logger.info(f"  ✓ Loaded from epoch {epoch}")
        logger.info(f"  ✓ Best S-measure during training: {best_metric}")
    
    # Case 2: Raw state_dict
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        logger.info(f"  ✓ Loaded state_dict")
    
    else:
        raise RuntimeError(f"Invalid checkpoint format")
    
    return model


# =========================================================
# Evaluation Metrics
# =========================================================
class MetricsAccumulator:
    """Accumulate metrics across batches"""
    def __init__(self):
        self.metrics = {
            "S": [],      # S-measure (Structure)
            "Fw": [],     # Weighted F-measure
            "E": [],      # E-measure (Enhanced-alignment)
            "MAE": []     # Mean Absolute Error
        }
    
    def update(self, pred, target):
        """
        Update metrics for a single sample
        
        Args:
            pred: Predicted probability map (1, H, W) or (H, W)
            target: Ground truth mask (1, H, W) or (H, W)
        """
        self.metrics["S"].append(s_measure(pred, target).item())
        self.metrics["E"].append(e_measure(pred, target).item())
        self.metrics["Fw"].append(fw_measure(pred, target).item())
        
        # MAE
        mae = torch.abs(pred - target).mean().item()
        self.metrics["MAE"].append(mae)
    
    def get_summary(self):
        """Get mean and std of all metrics"""
        summary = {}
        for key, values in self.metrics.items():
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
        return summary
    
    def get_detailed(self):
        """Get all individual values"""
        return self.metrics


# =========================================================
# Evaluation Function
# =========================================================
@torch.no_grad()
def evaluate_model(model, dataloader, device, logger):
    """
    Evaluate DualStream_PVT_COD model on a dataset
    
    Args:
        model: DualStream_PVT_COD
        dataloader: DataLoader returning (rgb, depth, mask)
        device: cuda or cpu
        logger: Logger
    
    Returns:
        accumulator: MetricsAccumulator with results
    """
    model.eval()
    accumulator = MetricsAccumulator()
    
    logger.info("Starting evaluation...")
    logger.info("Model: DualStream_PVT_COD (Deep Supervision)")
    logger.info("Using: pred_d1 (final output) for metrics")
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Evaluating", ncols=100)
    
    for rgb, depth, masks in pbar:
        rgb = rgb.to(device)
        depth = depth.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions, boundary = model(rgb, depth)
        pred_d1, pred_d2, pred_d3, pred_d4 = predictions
        
        # Use final prediction (d1) for evaluation
        pred_logits = pred_d1
        
        # Convert to probabilities
        pred_probs = torch.sigmoid(pred_logits)
        
        # Compute metrics for each sample in batch
        batch_size = rgb.shape[0]
        for i in range(batch_size):
            accumulator.update(pred_probs[i], masks[i])
        
        # Update progress bar with current average
        current_metrics = accumulator.get_summary()
        pbar.set_postfix({
            'S': f"{current_metrics['S_mean']:.4f}",
            'E': f"{current_metrics['E_mean']:.4f}",
            'MAE': f"{current_metrics['MAE_mean']:.4f}"
        })
    
    return accumulator


# =========================================================
# Results Display and Saving
# =========================================================
def display_results(summary, logger):
    """Display evaluation results in a nice format"""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    metrics_order = ['S', 'Fw', 'E', 'MAE']
    
    print("\nMetrics (Mean ± Std):")
    print("-" * 80)
    for metric in metrics_order:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        print(f"  {metric:8s}: {mean:.4f} ± {std:.4f}")
    
    print("-" * 80)
    
    # Log to file
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    for metric in metrics_order:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        logger.info(f"{metric:12s}: {mean:.4f} ± {std:.4f}")
    logger.info("="*80)
    
    print("\n" + "="*80 + "\n")


def save_results(summary, detailed, config):
    """Save evaluation results to JSON files"""
    
    # Save summary
    summary_path = os.path.join(config.save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = os.path.join(config.save_dir, "detailed.json")
    detailed_for_json = {
        k: [float(v) for v in v_list] 
        for k, v_list in detailed.items()
    }
    with open(detailed_path, 'w') as f:
        json.dump(detailed_for_json, f, indent=4)
    print(f"✓ Detailed results saved to: {detailed_path}")
    
    # Save config
    config_dict = {
        "model": "DualStream_PVT_COD",
        "checkpoint": config.ckpt_path,
        "dataset_root": config.root,
        "split": config.split,
        "img_size": config.img_size,
        "batch_size": config.batch_size,
        "use_depth": config.use_depth,
        "is_bem": config.is_bem,
        "is_cbam_en3": config.is_cbam_en3,
        "is_cbam_en4": config.is_cbam_en4,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = os.path.join(config.save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Config saved to: {config_path}")


# =========================================================
# Main Evaluation
# =========================================================
def main():
    """Main evaluation function"""
    
    # Initialize config
    config = EvalConfig()
    
    # Setup logger
    logger = setup_logger(config.save_dir, "evaluation.log")
    
    # Log configuration
    logger.info("="*80)
    logger.info("DUALSTREAM PVT-COD EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: DualStream_PVT_COD")
    logger.info(f"  - RGB Backbone: PVT-v2-b2")
    logger.info(f"  - Depth Backbone: PVT-v2-b0")
    logger.info(f"  - BFM: Enabled")
    logger.info(f"  - CBAM (e3, e4): {config.is_cbam_en3}, {config.is_cbam_en4}")
    logger.info(f"  - BEM: {config.is_bem}")
    logger.info(f"Checkpoint   : {config.ckpt_path}")
    logger.info(f"Dataset Root : {config.root}")
    logger.info(f"Split        : {config.split}")
    logger.info(f"Image Size   : {config.img_size}")
    logger.info(f"Use Depth    : {config.use_depth}")
    logger.info(f"Batch Size   : {config.batch_size}")
    logger.info(f"Device       : {config.device}")
    logger.info(f"Save Dir     : {config.save_dir}")
    logger.info("="*80)
    
    # Create model
    logger.info("\nCreating model...")
    model = DualStream_PVT_COD(
        n_classes=config.n_classes,
        is_bem=config.is_bem,
        is_cbam_en3=config.is_cbam_en3,
        is_cbam_en4=config.is_cbam_en4,
        pretrained=config.pretrained
    ).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Load checkpoint
    model = load_checkpoint(model, config.ckpt_path, config.device, logger)
    
    # Create dataset
    logger.info(f"\nLoading dataset: {config.split}")
    dataset = MHCDDatasetWithDepth(
        root=config.root,
        split=config.split,
        img_size=config.img_size,
        augment=False,  # No augmentation for evaluation
        use_depth=config.use_depth,
        logger=logger
    )
    logger.info(f"Total samples: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("STARTING EVALUATION")
    logger.info("="*80 + "\n")
    
    accumulator = evaluate_model(model, dataloader, config.device, logger)
    
    # Get results
    summary = accumulator.get_summary()
    detailed = accumulator.get_detailed()
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*80)
    display_results(summary, logger)
    
    # Save results
    save_results(summary, detailed, config)
    
    logger.info(f"\nAll results saved to: {config.save_dir}")
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"S-measure : {summary['S_mean']:.4f}")
    print(f"Fw-measure: {summary['Fw_mean']:.4f}")
    print(f"E-measure : {summary['E_mean']:.4f}")
    print(f"MAE       : {summary['MAE_mean']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()