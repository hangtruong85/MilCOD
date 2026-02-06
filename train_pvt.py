"""
Training script with Deep Supervision Loss
Implements loss = loss_d4 + loss_d3 + loss_d2 + loss_d1 + 4×loss_boundary
FIXED: With validation loss computation and train+val loss plotting
"""

import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime

from datasets.mhcd_dataset import MHCDDataset
from model_registry import create_model

from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.fweighted_measure import fw_measure
from metrics.mae_metric import mae_metric
from utils.logger import setup_logger
from utils.plot import plot_training_curves
from loss.deep_supervision_loss import DeepSupervisionLoss


# ===================== Configuration =====================

class Config:
    """Configuration for training with deep supervision"""
    def __init__(self):
        # Model selection
        self.model_name = "PVT_DeepSupervision_BEMMulti"
        
        # Dataset
        self.root = "../MHCD_seg"
        self.img_size = 352
        
        # Training
        self.epochs = 120
        self.batch_size = 12
        self.num_workers = 4
        
        # Optimizer
        self.lr_encoder = 1e-5
        self.lr_decoder = 1e-4
        self.weight_decay = 1e-4
        
        # Deep Supervision Loss Weights
        self.lambda_wbce = 0.5          # BCE weight in supervision loss
        self.lambda_wiou = 0.5           # IoU weight in supervision loss
        self.lambda_boundary = 4.0      # Boundary loss multiplier

        # Training strategy
        self.warmup_epochs = 5
        self.warmup_boundary_epochs = 30
        self.use_cosine_schedule = True
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Logging
        self.log_dir = f"logs/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)


# ===================== Training Functions =====================

def train_epoch(model, loader, optimizer, criterion, scaler, device, config, epoch):
    """
    Train for one epoch with deep supervision
    
    Args:
        model: Model with deep supervision heads
        loader: DataLoader
        optimizer: Optimizer
        criterion: DeepSupervisionLoss
        scaler: GradScaler for AMP
        device: cuda/cpu
        config: Config object
        epoch: Current epoch number
    """
    model.train()
    
    # Freeze encoder during warmup
    if epoch <= config.warmup_epochs:
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
    else:
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = True
    
    total_loss = 0.0
    num_batches = 0
    
    # Check if model supports boundary prediction
    has_boundary = hasattr(model, 'predict_boundary') and model.predict_boundary
    
    # Determine if we should use boundary loss
    use_boundary_loss = (epoch > config.warmup_boundary_epochs) and has_boundary
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass with all levels
            predictions = model(images, return_all_levels=True, return_boundary=True)
            
            if use_boundary_loss:
                (pred_d1, pred_d2, pred_d3, pred_d4), boundary_pred = predictions
                
                # Extract ground truth boundary
                from models.boundary_enhancement import BoundaryEnhancementModule
                bem_temp = BoundaryEnhancementModule(channels=1).to(device)
                boundary_target = bem_temp.extract_boundary_map(masks)
                
                # Normalize boundary_target
                boundary_target = torch.clamp(boundary_target, 0, 1)
                
                # Compute loss with boundary
                loss = criterion(pred_d1, pred_d2, pred_d3, pred_d4, 
                               masks, boundary_pred, boundary_target)
            else:
                (pred_d1, pred_d2, pred_d3, pred_d4), _ = predictions
                
                # Compute loss without boundary
                loss = criterion(pred_d1, pred_d2, pred_d3, pred_d4, 
                               masks, None, None)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


# ===================== Validation Functions =====================

@torch.no_grad()
def validate(model, loader, criterion, device, config):
    """
    Validate model using final prediction (d1)
    Computes both validation loss and metrics
    """
    model.eval()
    
    metrics = {
        "loss": 0.0,
        "S": 0.0,
        "E": 0.0,
        "Fw": 0.0,
        "MAE": 0.0
    }
    
    num_samples = 0
    has_boundary = hasattr(model, 'predict_boundary') and model.predict_boundary
    
    # Determine if we should use boundary loss in validation
    # Always use boundary during validation if model supports it
    use_boundary_loss = has_boundary
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass - get all levels for loss computation
        predictions = model(images, return_all_levels=True, return_boundary=True)
        
        if use_boundary_loss:
            (pred_d1, pred_d2, pred_d3, pred_d4), boundary_pred = predictions
            
            # Extract ground truth boundary
            from models.boundary_enhancement import BoundaryEnhancementModule
            bem_temp = BoundaryEnhancementModule(channels=1).to(device)
            boundary_target = bem_temp.extract_boundary_map(masks)
            
            # Normalize boundary_target
            boundary_target = torch.clamp(boundary_target, 0, 1)
            
            # Compute loss with boundary
            val_loss = criterion(pred_d1, pred_d2, pred_d3, pred_d4, 
                                masks, boundary_pred, boundary_target)
        else:
            (pred_d1, pred_d2, pred_d3, pred_d4), _ = predictions
            
            # Compute loss without boundary
            val_loss = criterion(pred_d1, pred_d2, pred_d3, pred_d4, 
                                masks, None, None)
        
        # Convert to probabilities (use d1 for metrics)
        pred_probs = torch.sigmoid(pred_d1)
        
        # Compute metrics for each sample
        batch_size = images.shape[0]
        for i in range(batch_size):
            metrics["S"] += s_measure(pred_probs[i], masks[i]).item()
            metrics["E"] += e_measure(pred_probs[i], masks[i]).item()
            metrics["Fw"] += fw_measure(pred_probs[i], masks[i]).item()
            metrics["MAE"] += mae_metric(pred_d1[i:i+1], masks[i:i+1]).item()
        
        metrics["loss"] += val_loss.item() * batch_size
        num_samples += batch_size
    
    # Average metrics
    for key in metrics:
        metrics[key] /= num_samples
    
    return metrics


# ===================== Optimizer Creation =====================

def create_optimizer(model, config):
    """Create optimizer with different learning rates"""
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'backbone' in name or 'cbam' in name.lower():
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    param_groups = []
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': config.lr_encoder,
            'name': 'encoder'
        })
        print(f"  Encoder params: {len(encoder_params)} tensors, LR={config.lr_encoder}")
    
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': config.lr_decoder,
            'name': 'decoder'
        })
        print(f"  Decoder params: {len(decoder_params)} tensors, LR={config.lr_decoder}")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay
    )
    
    return optimizer


def create_scheduler(optimizer, config, num_batches):
    """Create learning rate scheduler"""
    if config.use_cosine_schedule:
        total_steps = config.epochs * num_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5
        )
    
    return scheduler


# ===================== Main Training Loop =====================

def main():
    """Main training function"""
    
    # Initialize configuration
    config = Config()
    
    # Setup logger
    logger = setup_logger(config.log_dir, "train.log")
    
    # Log configuration
    logger.info("="*80)
    logger.info("DEEP SUPERVISION TRAINING (WITH VALIDATION LOSS)")
    logger.info("="*80)
    logger.info(f"Model Name    : {config.model_name}")
    logger.info(f"Dataset Root  : {config.root}")
    logger.info(f"Image Size    : {config.img_size}")
    logger.info(f"Batch Size    : {config.batch_size}")
    logger.info(f"Epochs        : {config.epochs}")
    logger.info(f"Warmup Epochs : {config.warmup_epochs}")
    logger.info(f"Warmup Boundary Epochs: {config.warmup_boundary_epochs}")
    logger.info(f"LR Encoder    : {config.lr_encoder}")
    logger.info(f"LR Decoder    : {config.lr_decoder}")
    logger.info(f"Supervision Loss Weights:")
    logger.info(f"  wBCE: {config.lambda_wbce}, wIoU: {config.lambda_wiou}")
    logger.info(f"  Boundary multiplier: {config.lambda_boundary}")
    logger.info(f"Device        : {config.device}")
    logger.info(f"Log Directory : {config.log_dir}")
    logger.info("="*80)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = MHCDDataset(config.root, "train", config.img_size, logger=logger)
    val_dataset = MHCDDataset(config.root, "val", config.img_size, logger=logger)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples  : {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info(f"Creating model: {config.model_name}")
    model = create_model(config.model_name, config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters    : {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss, optimizer, scheduler
    criterion = DeepSupervisionLoss(
        lambda_wbce=config.lambda_wbce,
        lambda_wiou=config.lambda_wiou,
        lambda_boundary=config.lambda_boundary
    )
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler()
    
    # Training state
    start_epoch = 1
    best_s_measure = 0.0
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(config.log_dir, "last.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_s_measure = checkpoint["best_s_measure"]
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        val_metrics_history = checkpoint.get("val_metrics", [])
        
        logger.info(f"Resumed from epoch {start_epoch}, best S-measure: {best_s_measure:.4f}")
    
    # CSV logging
    csv_path = os.path.join(config.log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "S_measure", "E_measure", "fw_measure", "MAE"])
    
    # Training loop
    logger.info("="*80)
    logger.info("STARTING TRAINING WITH DEEP SUPERVISION")
    logger.info("="*80)
    
    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch}/{config.epochs}")
        logger.info(f"{'='*80}")
        
        # Training phase
        if epoch <= config.warmup_epochs:
            logger.info(f"[WARMUP] Encoder frozen")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                scaler, config.device, config, epoch)
        train_losses.append(train_loss)
        
        # Validation phase (with loss computation)
        val_metrics = validate(model, val_loader, criterion, config.device, config)
        val_losses.append(val_metrics["loss"])
        val_metrics_history.append(val_metrics)
        
        # Logging
        logger.info(f"[TRAIN] Loss: {train_loss:.4f}")
        logger.info(f"[VAL]   Loss: {val_metrics['loss']:.4f} | "
                   f"S: {val_metrics['S']:.4f} | "
                   f"E: {val_metrics['E']:.4f} | "
                   f"F: {val_metrics['Fw']:.4f} | "
                   f"MAE: {val_metrics['MAE']:.4f}")
        
        # Save to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["S"],
                val_metrics["E"],
                val_metrics["Fw"],
                val_metrics["MAE"]
            ])
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_s_measure": best_s_measure,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics_history,
        }
        torch.save(checkpoint, os.path.join(config.log_dir, "last.pth"))
        
        # Save best model
        if val_metrics["S"] > best_s_measure:
            best_s_measure = val_metrics["S"]
            torch.save(model.state_dict(), os.path.join(config.log_dir, "best_s_measure.pth"))
            logger.info(f"🔥 NEW BEST S-measure: {best_s_measure:.4f}")
        
        # Update learning rate
        if config.use_cosine_schedule:
            for _ in range(len(train_loader)):
                scheduler.step()
        else:
            scheduler.step()
        
        # Plot training curves (WITH BOTH TRAIN AND VAL LOSS)
        if epoch % 5 == 0 or epoch == config.epochs:
            plot_training_curves(train_losses, val_losses, val_metrics_history, config.log_dir)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Best S-measure: {best_s_measure:.4f}")
    logger.info(f"Model saved to: {config.log_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()