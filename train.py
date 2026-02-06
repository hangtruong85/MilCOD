"""
Training script for DualStream_PVT_COD
Simplified version - only supports dual-stream RGB+Depth model
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
from models.dual_stream_pvt import DualStream_PVT_COD

from metrics.s_measure_paper import s_measure
from metrics.e_measure_paper import e_measure
from metrics.fweighted_measure import fw_measure
from metrics.mae_metric import mae_metric
from utils.logger import setup_logger
from utils.plot import plot_training_curves
from loss.deep_supervision_loss import DeepSupervisionLoss


# ===================== Configuration =====================

class Config:
    """Configuration for DualStream_PVT_COD training"""
    def __init__(self):
        # Model settings
        self.n_classes = 1
        self.is_bem = True           # Use Boundary Enhancement Module
        self.is_cbam_en3 = True      # Use CBAM on encoder stage 3
        self.is_cbam_en4 = True      # Use CBAM on encoder stage 4
        self.pretrained = True       # Use pretrained PVT backbones
        
        # Dataset
        self.root = "../MHCD_seg"
        self.img_size = 352
        self.use_depth = True        # Load depth maps
        
        # Training
        self.epochs = 120
        self.batch_size = 12
        self.num_workers = 4
        
        # Optimizer
        self.lr_encoder = 1e-5       # Learning rate for encoders (RGB + Depth)
        self.lr_decoder = 1e-4       # Learning rate for decoder/BFM/BEM
        self.weight_decay = 1e-4
        
        # Deep Supervision Loss Weights
        self.lambda_wbce = 0.5          # BCE weight
        self.lambda_wiou = 0.5          # IoU weight
        self.lambda_boundary = 4.0      # Boundary loss multiplier

        # Training strategy
        self.warmup_epochs = 5           # Freeze encoder for first N epochs
        self.warmup_boundary_epochs = 20 # Start boundary loss after N epochs
        self.use_cosine_schedule = True  # Use cosine LR schedule
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"logs/DualStream_PVT_COD_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)


# ===================== Training Functions =====================

def train_epoch(model, loader, optimizer, criterion, scaler, device, config, epoch):
    """
    Train for one epoch with deep supervision
    
    Args:
        model: DualStream_PVT_COD
        loader: DataLoader returning (rgb, depth, mask)
        optimizer: Optimizer
        criterion: DeepSupervisionLoss
        scaler: GradScaler for AMP
        device: cuda/cpu
        config: Config object
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average training loss
    """
    model.train()
    
    # Freeze encoder during warmup
    if epoch <= config.warmup_epochs:
        for param in model.dual_encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.dual_encoder.parameters():
            param.requires_grad = True
    
    total_loss = 0.0
    num_batches = 0
    
    # Determine if we should use boundary loss
    use_boundary_loss = (epoch > config.warmup_boundary_epochs) and config.is_bem
    
    for batch_idx, (rgb, depth, masks) in enumerate(loader):
        rgb = rgb.to(device)
        depth = depth.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass
            predictions, boundary_pred = model(rgb, depth)
            pred_d1, pred_d2, pred_d3, pred_d4 = predictions
            
            if use_boundary_loss and boundary_pred is not None:
                # Extract ground truth boundary
                from models.boundary_enhancement import BoundaryEnhancementModule
                bem_temp = BoundaryEnhancementModule(channels=1).to(device)
                boundary_target = bem_temp.extract_boundary_map(masks)
                boundary_target = torch.clamp(boundary_target, 0, 1)
                
                # Compute loss with boundary
                loss = criterion(pred_d1, pred_d2, pred_d3, pred_d4, 
                               masks, boundary_pred, boundary_target)
            else:
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
    Validate model
    Computes validation loss and metrics (S, E, Fw, MAE)
    
    Args:
        model: DualStream_PVT_COD
        loader: DataLoader returning (rgb, depth, mask)
        criterion: DeepSupervisionLoss
        device: cuda/cpu
        config: Config object
    
    Returns:
        metrics: Dict with loss, S, E, Fw, MAE
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
    use_boundary_loss = config.is_bem
    
    for rgb, depth, masks in loader:
        rgb = rgb.to(device)
        depth = depth.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions, boundary_pred = model(rgb, depth)
        pred_d1, pred_d2, pred_d3, pred_d4 = predictions
        
        if use_boundary_loss and boundary_pred is not None:
            # Extract ground truth boundary
            from models.boundary_enhancement import BoundaryEnhancementModule
            bem_temp = BoundaryEnhancementModule(channels=1).to(device)
            boundary_target = bem_temp.extract_boundary_map(masks)
            boundary_target = torch.clamp(boundary_target, 0, 1)
            
            # Compute loss with boundary
            val_loss = criterion(pred_d1, pred_d2, pred_d3, pred_d4, 
                                masks, boundary_pred, boundary_target)
        else:
            # Compute loss without boundary
            val_loss = criterion(pred_d1, pred_d2, pred_d3, pred_d4, 
                                masks, None, None)
        
        # Convert to probabilities (use d1 for metrics)
        pred_probs = torch.sigmoid(pred_d1)
        
        # Compute metrics for each sample
        batch_size = rgb.shape[0]
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
    """
    Create optimizer with different learning rates for encoder and decoder
    
    Encoder: RGB backbone + Depth backbone + BFM
    Decoder: Decoder blocks + CBAM + BEM + Output heads
    """
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Encoder: dual_encoder (RGB backbone, Depth backbone, BFM)
        if 'dual_encoder' in name:
            encoder_params.append(param)
        else:
            # Decoder: decoder blocks, CBAM, BEM, output heads
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
    logger.info("DUALSTREAM PVT-COD TRAINING")
    logger.info("="*80)
    logger.info(f"Model: DualStream_PVT_COD")
    logger.info(f"  - RGB Backbone: PVT-v2-b2")
    logger.info(f"  - Depth Backbone: PVT-v2-b0")
    logger.info(f"  - BFM (Bi-directional Fusion): Enabled")
    logger.info(f"  - CBAM (e3, e4): {config.is_cbam_en3}, {config.is_cbam_en4}")
    logger.info(f"  - BEM (Boundary): {config.is_bem}")
    logger.info(f"Dataset Root  : {config.root}")
    logger.info(f"Image Size    : {config.img_size}")
    logger.info(f"Use Depth     : {config.use_depth}")
    logger.info(f"Batch Size    : {config.batch_size}")
    logger.info(f"Epochs        : {config.epochs}")
    logger.info(f"Warmup Epochs : {config.warmup_epochs}")
    logger.info(f"Warmup Boundary: {config.warmup_boundary_epochs}")
    logger.info(f"LR Encoder    : {config.lr_encoder}")
    logger.info(f"LR Decoder    : {config.lr_decoder}")
    logger.info(f"Loss Weights:")
    logger.info(f"  wBCE: {config.lambda_wbce}, wIoU: {config.lambda_wiou}")
    logger.info(f"  Boundary: {config.lambda_boundary}")
    logger.info(f"Device        : {config.device}")
    logger.info(f"Log Directory : {config.log_dir}")
    logger.info("="*80)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = MHCDDataset(
        root=config.root,
        split="train",
        img_size=config.img_size,
        augment=True,
        use_depth=config.use_depth,
        logger=logger
    )
    
    val_dataset = MHCDDataset(
        root=config.root,
        split="val",
        img_size=config.img_size,
        augment=False,
        use_depth=config.use_depth,
        logger=logger
    )
    
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
    logger.info("Creating model...")
    model = DualStream_PVT_COD(
        n_classes=config.n_classes,
        is_bem=config.is_bem,
        is_cbam_en3=config.is_cbam_en3,
        is_cbam_en4=config.is_cbam_en4,
        pretrained=config.pretrained
    ).to(config.device)
    
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
            writer.writerow(["epoch", "train_loss", "val_loss", 
                           "S_measure", "E_measure", "fw_measure", "MAE"])
    
    # Training loop
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch}/{config.epochs}")
        logger.info(f"{'='*80}")
        
        # Training phase
        if epoch <= config.warmup_epochs:
            logger.info(f"[WARMUP] Encoder frozen")
        
        if epoch <= config.warmup_boundary_epochs:
            logger.info(f"[WARMUP] Boundary loss disabled")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            scaler, config.device, config, epoch
        )
        train_losses.append(train_loss)
        
        # Validation phase
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
        
        # Plot training curves
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
