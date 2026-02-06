import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from datasets.mhcd_dataset import MHCDDataset

from model_registry import create_model

from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.f_measure_paper import f_measure
from metrics.fweighted_measure import fw_measure
from utils.logger import setup_logger


# =========================================================
# Configuration
# =========================================================
class EvalConfig:
    """Evaluation configuration"""
    def __init__(self):
        # Model to evaluate
        self.model_name = "PVT_DeepSupervision_BEMMulti"
        
        # Checkpoint path
        self.ckpt_path = "logs/PVT_DeepSupervision_BEMMulti_20260206_102516/best_s_measure.pth"
        
        # Dataset
        self.root = "../MHCD_seg"
        self.split = "test"  # or "val"
        self.img_size = 352
        self.batch_size = 12
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Output
        self.save_dir = f"eval_results/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)

# =========================================================
# Checkpoint Loading
# =========================================================
def load_checkpoint(model, ckpt_path, device):
    """
    Smart checkpoint loading - handles different formats
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Case 1: Checkpoint is a dictionary with 'model' key
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "unknown")
        best_metric = ckpt.get("best_s_measure", "unknown")
        print(f"  ✓ Loaded from epoch {epoch}, best S-measure: {best_metric}")
    
    # Case 2: Checkpoint is raw state_dict
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        print(f"  ✓ Loaded state_dict")
    
    else:
        raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")
    
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
            #"F": [],      # F-measure (F-beta)
            "MAE": []     # Mean Absolute Error
        }
    
    def update(self, pred, target):
        """
        Update metrics for a single sample
        Args:
            pred: predicted probability map (C, H, W)
            target: ground truth mask (C, H, W)
        """
        self.metrics["S"].append(s_measure(pred, target).item())
        self.metrics["Fw"].append(fw_measure(pred, target).item())
        self.metrics["E"].append(e_measure(pred, target).item())
        #self.metrics["F"].append(f_measure(pred, target).item())
        self.metrics["MAE"].append(torch.abs(pred - target).mean().item())
    
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
    Evaluate model on a dataset
    
    Handles both:
    - Simple models: output is tensor (B, 1, H, W)
    - Deep Supervision models: output is tuple ((d1, d2, d3, d4), boundary) or (tensor, tensor)
    """
    model.eval()
    accumulator = MetricsAccumulator()
    
    logger.info("Starting evaluation...")
    
    # Detect if model is deep supervision
    is_deep_supervision = hasattr(model, 'out_d1')
    
    if is_deep_supervision:
        logger.info("Model type: Deep Supervision (using d1 output)")
    else:
        logger.info("Model type: Standard")
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Evaluating", ncols=100)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images, return_all_levels=False, return_boundary=False) \
                  if is_deep_supervision else model(images)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            # Deep supervision or boundary model
            # outputs = (pred, boundary) or similar
            pred_logits = outputs[0]
        else:
            # Standard model
            pred_logits = outputs
        
        # Convert to probabilities
        pred_probs = torch.sigmoid(pred_logits)
        
        # Compute metrics for each sample in batch
        batch_size = images.shape[0]
        for i in range(batch_size):
            accumulator.update(pred_probs[i], masks[i])
        
        # Update progress bar with current average
        current_metrics = accumulator.get_summary()
        pbar.set_postfix({
            'S': f"{current_metrics['S_mean']:.4f}",
            'MAE': f"{current_metrics['MAE_mean']:.4f}"
        })
    
    return accumulator


# =========================================================
# Results Display and Saving
# =========================================================
def display_results(summary, logger):
    """
    Display evaluation results in a nice format
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    metrics_order = ['S', 'Fw', 'E', 'MAE']
    
    print("\nConsole Output:")
    for metric in metrics_order:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        print(f"{metric:8s}: {mean:.4f} ± {std:.4f}")
    
    print("\nLog Output:")
    for metric in metrics_order:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        logger.info(f"{metric:12s}: {mean:.4f} ± {std:.4f}")
    
    print("="*60 + "\n")


def save_results(summary, detailed, config):
    """
    Save evaluation results to JSON files
    """
    # Save summary
    summary_path = os.path.join(config.save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = os.path.join(config.save_dir, "detailed.json")
    # Convert numpy arrays to lists for JSON serialization
    detailed_for_json = {k: [float(v) for v in v_list] for k, v_list in detailed.items()}
    with open(detailed_path, 'w') as f:
        json.dump(detailed_for_json, f, indent=4)
    print(f"✓ Detailed results saved to: {detailed_path}")
    
    # Save config
    config_dict = {
        "model_name": config.model_name,
        "checkpoint": config.ckpt_path,
        "dataset_root": config.root,
        "split": config.split,
        "img_size": config.img_size,
        "batch_size": config.batch_size,
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
    # Initialize config
    config = EvalConfig()
    
    # Setup logger
    logger = setup_logger(config.save_dir, "evaluation.log")
    
    logger.info("="*80)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model Name   : {config.model_name}")
    logger.info(f"Checkpoint   : {config.ckpt_path}")
    logger.info(f"Dataset Root : {config.root}")
    logger.info(f"Split        : {config.split}")
    logger.info(f"Image Size   : {config.img_size}")
    logger.info(f"Batch Size   : {config.batch_size}")
    logger.info(f"Device       : {config.device}")
    logger.info(f"Save Dir     : {config.save_dir}")
    logger.info("="*80)
    
    # Create model
    logger.info(f"\nCreating model: {config.model_name}")
    model = create_model(config.model_name, config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Load checkpoint
    model = load_checkpoint(model, config.ckpt_path, config.device)
    
    # Create dataset
    logger.info(f"\nLoading dataset: {config.split}")
    dataset = MHCDDataset(
        config.root,
        config.split,
        config.img_size,
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
    logger.info("="*80 + "\n")
    display_results(summary, logger)
    
    # Save results
    save_results(summary, detailed, config)
    
    logger.info(f"\nAll results saved to: {config.save_dir}")

if __name__ == "__main__":
    import sys
    
    # Evaluate single model
    main()