"""
Visualization tool for UNet3Plus_B3_BEM model
Visualizes: Encoder features, Decoder features (full-scale skip connections),
            Boundary Enhancement Module (BEM), Sobel boundary maps, and attention maps
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from datasets.mhcd_dataset import MHCDDataset
from models.unet3plus import UNet3Plus_B3_BEM


# ============================================================================
# Hook-based Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Extract intermediate features from model layers using forward hooks."""

    def __init__(self):
        self.features = {}
        self.hooks = []

    def register_hook(self, module, name):
        """Register a forward hook on a single module."""
        hook = module.register_forward_hook(self._make_hook(name))
        self.hooks.append(hook)

    def register_hooks_by_path(self, model, layer_paths):
        """
        Register hooks by dot-separated module paths.
        Args:
            model: nn.Module
            layer_paths: dict {display_name: "module.path.string"}
        """
        for name, path in layer_paths.items():
            module = model
            for attr in path.split('.'):
                if hasattr(module, attr):
                    module = getattr(module, attr)
                else:
                    print(f"  Warning: cannot find '{attr}' in path '{path}'")
                    module = None
                    break
            if module is not None:
                self.register_hook(module, name)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                self.features[name] = output[0].detach()
        return hook_fn

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ============================================================================
# Attention Map Utilities
# ============================================================================

def generate_cam(features, method='norm'):
    """
    Generate Class Activation Map from feature tensor.
    Args:
        features: (B, C, H, W)
        method: 'mean' | 'max' | 'norm'
    Returns:
        cam: (B, 1, H, W)
    """
    if method == 'mean':
        return features.mean(dim=1, keepdim=True)
    elif method == 'max':
        return features.max(dim=1, keepdim=True)[0]
    elif method == 'norm':
        return torch.norm(features, dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown CAM method: {method}")


def normalize_to_numpy(tensor):
    """Normalize a (H, W) or (1, H, W) tensor to [0,1] numpy array."""
    t = tensor.squeeze().cpu().float()
    t_min, t_max = t.min(), t.max()
    if t_max - t_min > 1e-8:
        t = (t - t_min) / (t_max - t_min)
    else:
        t = torch.zeros_like(t)
    return t.numpy()


def tensor_to_rgb(image_tensor):
    """Convert (3, H, W) normalized image tensor back to displayable [0,1] RGB."""
    img = image_tensor.cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


# ============================================================================
# UNet3Plus_B3_BEM Visualizer
# ============================================================================

class UNet3PlusBEMVisualizer:
    """
    Comprehensive visualization for UNet3Plus_B3_BEM.

    Generates a multi-row figure:
        Row 1: Input | Ground Truth | Prediction | Overlay | Error Map
        Row 2: Encoder features (e1 .. e5)
        Row 3: Decoder features (d4, d3, d2, d1) + BEM output
        Row 4: BEM analysis (edge features, Sobel boundary, boundary pred, overlay)
        Row 5: Attention maps (CAM) at different decoder levels
    """

    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.extractor = FeatureExtractor()

    # ------------------------------------------------------------------
    # Hook registration helpers
    # ------------------------------------------------------------------

    def _register_encoder_hooks(self):
        """Register hooks on encoder stages to capture e1..e5."""
        encoder = self.model.encoder
        # smp encoder stages are accessible via encoder._blocks or encoder.model
        # The simplest way: hook into the encoder's forward and capture the list output
        # Instead we hook the whole encoder and intercept its return value
        def encoder_hook(module, input, output):
            # output is a list [e0, e1, e2, e3, e4, e5]
            if isinstance(output, (list, tuple)):
                for i, feat in enumerate(output):
                    if isinstance(feat, torch.Tensor):
                        self.extractor.features[f'e{i}'] = feat.detach()
        hook = encoder.register_forward_hook(encoder_hook)
        self.extractor.hooks.append(hook)

    def _register_decoder_hooks(self):
        """Register hooks on decoder blocks to capture d1..d4."""
        for name in ['decoder4', 'decoder3', 'decoder2', 'decoder1']:
            module = getattr(self.model, name, None)
            if module is not None:
                self.extractor.register_hook(module, name.replace('decoder', 'd'))

    def _register_bem_hooks(self):
        """Register hooks on BEM sub-modules."""
        bem = self.model.bem

        # Edge conv output
        self.extractor.register_hook(bem.edge_conv, 'bem_edge_conv')

        # Fusion output
        self.extractor.register_hook(bem.fusion, 'bem_fusion')

        # Boundary head (if exists)
        if hasattr(bem, 'boundary_head'):
            self.extractor.register_hook(bem.boundary_head, 'bem_boundary_head')

    # ------------------------------------------------------------------
    # Forward with hooks
    # ------------------------------------------------------------------

    def _forward_with_hooks(self, image):
        """Run forward pass with all hooks registered, return (pred_logit, boundary_logit)."""
        self.extractor.clear()
        self.extractor.remove_hooks()

        self._register_encoder_hooks()
        self._register_decoder_hooks()
        self._register_bem_hooks()

        with torch.no_grad():
            output = self.model(image, return_boundary=True)

        if isinstance(output, tuple):
            pred_logit, boundary_logit = output
        else:
            pred_logit = output
            boundary_logit = None

        return pred_logit, boundary_logit

    # ------------------------------------------------------------------
    # Individual row renderers
    # ------------------------------------------------------------------

    def _draw_row_basic(self, axes, img_np, mask_np, pred_np):
        """Row 1: Input, GT, Prediction, Overlay, Error Map."""
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image', fontsize=10, fontweight='bold')

        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Ground Truth', fontsize=10, fontweight='bold')

        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title('Prediction', fontsize=10, fontweight='bold')

        axes[3].imshow(img_np)
        axes[3].imshow(pred_np, cmap='jet', alpha=0.5)
        axes[3].set_title('Overlay', fontsize=10, fontweight='bold')

        error = np.abs(mask_np - pred_np)
        im = axes[4].imshow(error, cmap='hot', vmin=0, vmax=1)
        axes[4].set_title(f'Error Map (MAE={error.mean():.4f})', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=axes[4], fraction=0.046)

    def _draw_row_encoder(self, axes):
        """Row 2: Encoder features e1..e5."""
        features = self.extractor.features
        for i in range(5):
            key = f'e{i+1}'
            ax = axes[i]
            if key in features:
                feat = features[key]
                cam = generate_cam(feat, method='norm')
                cam_np = normalize_to_numpy(cam[0])
                im = ax.imshow(cam_np, cmap='jet')
                ax.set_title(f'{key} ({feat.shape[1]}ch, {feat.shape[2]}x{feat.shape[3]})',
                             fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, f'{key}\nnot found', ha='center', va='center', fontsize=9)

    def _draw_row_decoder(self, axes):
        """Row 3: Decoder features d4, d3, d2, d1 + BEM fusion output."""
        features = self.extractor.features
        decoder_keys = ['d4', 'd3', 'd2', 'd1']

        for i, key in enumerate(decoder_keys):
            ax = axes[i]
            if key in features:
                feat = features[key]
                cam = generate_cam(feat, method='norm')
                cam_np = normalize_to_numpy(cam[0])
                im = ax.imshow(cam_np, cmap='jet')
                ax.set_title(f'{key} ({feat.shape[1]}ch, {feat.shape[2]}x{feat.shape[3]})',
                             fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, f'{key}\nnot found', ha='center', va='center', fontsize=9)

        # Column 5: BEM fusion
        ax = axes[4]
        if 'bem_fusion' in features:
            feat = features['bem_fusion']
            cam = generate_cam(feat, method='norm')
            cam_np = normalize_to_numpy(cam[0])
            im = ax.imshow(cam_np, cmap='jet')
            ax.set_title(f'BEM Fusion ({feat.shape[1]}ch)', fontsize=9, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, 'BEM Fusion\nnot found', ha='center', va='center', fontsize=9)

    def _draw_row_bem(self, axes, img_np, mask_np, pred_np, boundary_logit):
        """Row 4: BEM analysis -- edge features, Sobel boundary (from GT), boundary pred, overlays."""
        features = self.extractor.features

        # Col 0: BEM edge conv features
        ax = axes[0]
        if 'bem_edge_conv' in features:
            feat = features['bem_edge_conv']
            cam = generate_cam(feat, method='norm')
            cam_np = normalize_to_numpy(cam[0])
            im = ax.imshow(cam_np, cmap='inferno')
            ax.set_title('BEM Edge Features', fontsize=9, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, 'Edge features\nnot found', ha='center', va='center', fontsize=9)

        # Col 1: Sobel boundary from GT mask
        ax = axes[1]
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
        sobel_boundary = self.model.bem.extract_boundary_map(mask_t)
        sobel_np = normalize_to_numpy(sobel_boundary[0])
        im = ax.imshow(sobel_np, cmap='hot')
        ax.set_title('GT Boundary (Sobel)', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Col 2: Boundary prediction
        ax = axes[2]
        if boundary_logit is not None:
            boundary_pred = torch.sigmoid(boundary_logit)
            boundary_np = boundary_pred[0, 0].cpu().numpy()
            im = ax.imshow(boundary_np, cmap='hot')
            ax.set_title('Boundary Prediction', fontsize=9, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
        elif 'bem_boundary_head' in features:
            feat = features['bem_boundary_head']
            boundary_np = torch.sigmoid(feat[0, 0]).cpu().numpy()
            im = ax.imshow(boundary_np, cmap='hot')
            ax.set_title('Boundary Prediction', fontsize=9, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, 'No boundary\nprediction', ha='center', va='center', fontsize=9)
            boundary_np = None

        # Col 3: Boundary overlay on image
        ax = axes[3]
        ax.imshow(img_np)
        if boundary_logit is not None:
            boundary_overlay = torch.sigmoid(boundary_logit)[0, 0].cpu().numpy()
            ax.imshow(boundary_overlay, cmap='Reds', alpha=0.6)
        ax.set_title('Boundary Overlay', fontsize=9, fontweight='bold')

        # Col 4: Prediction contour on image
        ax = axes[4]
        ax.imshow(img_np)
        # Draw prediction contour and GT contour
        ax.contour(mask_np, levels=[0.5], colors='lime', linewidths=1.5)
        ax.contour(pred_np, levels=[0.5], colors='red', linewidths=1.5)
        ax.set_title('Contours (Green=GT, Red=Pred)', fontsize=9, fontweight='bold')

    def _draw_row_attention(self, axes, image):
        """Row 5: Attention maps (Grad-CAM style) from decoder levels."""
        features = self.extractor.features
        # Show CAM with different methods for different decoder levels
        methods = [('d4', 'mean'), ('d3', 'mean'), ('d2', 'max'), ('d1', 'max'), ('bem_fusion', 'norm')]
        labels = ['d4 (mean)', 'd3 (mean)', 'd2 (max)', 'd1 (max)', 'BEM (norm)']

        img_size = image.shape[2:]  # (H, W)

        for i, ((key, method), label) in enumerate(zip(methods, labels)):
            ax = axes[i]
            if key in features:
                feat = features[key]
                cam = generate_cam(feat, method=method)
                # Upsample to input resolution
                cam_up = F.interpolate(cam, size=img_size, mode='bilinear', align_corners=False)
                cam_np = normalize_to_numpy(cam_up[0])

                # Overlay on input image
                img_np = tensor_to_rgb(image[0])
                ax.imshow(img_np)
                ax.imshow(cam_np, cmap='jet', alpha=0.5)
                ax.set_title(f'CAM: {label}', fontsize=9)
            else:
                ax.text(0.5, 0.5, f'{key}\nnot found', ha='center', va='center', fontsize=9)

    # ------------------------------------------------------------------
    # Main visualization entry
    # ------------------------------------------------------------------

    def visualize_sample(self, image, mask, save_path):
        """
        Create full visualization for one sample.
        Args:
            image: (1, 3, H, W) tensor
            mask: (1, 1, H, W) tensor
            save_path: output image path
        """
        image = image.to(self.device)
        mask = mask.to(self.device)

        # Forward pass with hooks
        pred_logit, boundary_logit = self._forward_with_hooks(image)
        pred_prob = torch.sigmoid(pred_logit)

        # Convert to numpy
        img_np = tensor_to_rgb(image[0])
        mask_np = mask[0, 0].cpu().numpy()
        pred_np = pred_prob[0, 0].cpu().numpy()

        # Create 5-row x 5-col figure
        fig = plt.figure(figsize=(22, 22))
        gs = GridSpec(5, 5, figure=fig, hspace=0.35, wspace=0.3)

        # Build axes grid
        axes = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(5)]

        # Row 0: Basic results
        self._draw_row_basic(axes[0], img_np, mask_np, pred_np)

        # Row 1: Encoder features
        self._draw_row_encoder(axes[1])

        # Row 2: Decoder features + BEM fusion
        self._draw_row_decoder(axes[2])

        # Row 3: BEM analysis
        self._draw_row_bem(axes[3], img_np, mask_np, pred_np, boundary_logit)

        # Row 4: Attention maps (CAM overlay)
        self._draw_row_attention(axes[4], image)

        # Turn off all axes ticks
        for row in axes:
            for ax in row:
                ax.axis('off')

        # Row labels on the left
        row_labels = [
            'Predictions',
            'Encoder Features (e1-e5)',
            'Decoder Features (d4-d1) + BEM',
            'Boundary Enhancement Module',
            'Attention Maps (CAM)',
        ]
        for r, label in enumerate(row_labels):
            axes[r][0].annotate(
                label, xy=(-0.15, 0.5), xycoords='axes fraction',
                fontsize=11, fontweight='bold', rotation=90,
                ha='center', va='center',
            )

        fig.suptitle('UNet3Plus_B3_BEM  --  Visualization', fontsize=16, fontweight='bold', y=0.98)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Cleanup
        self.extractor.clear()
        self.extractor.remove_hooks()

        print(f"  Saved: {save_path}")


# ============================================================================
# Batch Visualization (multiple samples)
# ============================================================================

def visualize_batch(model, dataset, indices, save_dir, device):
    """Visualize multiple samples from the dataset."""
    vis = UNet3PlusBEMVisualizer(model, device)

    for idx in indices:
        image, mask = dataset[idx]
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        save_path = os.path.join(save_dir, f"sample_{idx:04d}.png")
        print(f"Processing sample {idx} ...")
        vis.visualize_sample(image, mask, save_path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize UNet3Plus_B3_BEM features and attention")
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--root', type=str, default='../MHCD_seg',
                        help='Dataset root directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--img_size', type=int, default=352,
                        help='Input image size')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='Directory to save output images')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("UNet3Plus_B3_BEM  --  Feature & Attention Visualization")
    print("=" * 70)

    # --- Create model ---
    print(f"\nCreating UNet3Plus_B3_BEM model ...")
    model = UNet3Plus_B3_BEM(n_classes=1, predict_boundary=True)

    # --- Load checkpoint ---
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            epoch = ckpt.get('epoch', '?')
            best = ckpt.get('best_s_measure', '?')
            print(f"  Loaded from epoch {epoch}, best S-measure: {best}")
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
            print(f"  Loaded state_dict")
        else:
            print(f"  Warning: unrecognized checkpoint format")
    else:
        print("  No checkpoint provided -- using random weights (for structure testing)")

    model = model.to(device).eval()

    # --- Load dataset ---
    print(f"\nLoading dataset: {args.root} [{args.split}] ...")
    dataset = MHCDDataset(args.root, args.split, args.img_size)
    print(f"  Total samples: {len(dataset)}")

    # --- Select samples ---
    num = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num, dtype=int)

    # --- Visualize ---
    save_dir = os.path.join(args.save_dir, 'UNet3Plus_B3_BEM')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nVisualizing {num} samples -> {save_dir}/\n")

    visualize_batch(model, dataset, indices, save_dir, device)

    print("\n" + "=" * 70)
    print(f"Done. All visualizations saved to: {save_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
