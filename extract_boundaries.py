"""
Extract boundary maps from all training masks
Uses BoundaryEnhancementModule.extract_boundary_map()
Saves boundary maps to 'boundaries' subfolder in train directory
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Import BEM
from models.boundary_enhancement import BoundaryEnhancementModule


class BoundaryExtractor:
    """Extract and save boundary maps from mask images"""
    
    def __init__(self, device='cuda'):
        """
        Initialize boundary extractor
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.bem = BoundaryEnhancementModule(channels=1).to(device)
        print(f"[INFO] BoundaryEnhancementModule initialized on {device}")
    
    def load_mask(self, mask_path):
        """
        Load mask image
        
        Args:
            mask_path: Path to mask image
            
        Returns:
            torch tensor (1, 1, H, W) with values in [0, 1]
        """
        # Read image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Convert to [0, 1]
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0
        
        # Convert to tensor (1, 1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return mask_tensor
    
    def extract_boundary(self, mask_tensor):
        """
        Extract boundary map from mask
        
        Args:
            mask_tensor: torch tensor (1, 1, H, W)
            
        Returns:
            boundary_map: numpy array (H, W) with values in [0, 1]
        """
        with torch.no_grad():
            boundary_tensor = self.bem.extract_boundary_map(mask_tensor)
            
            # Normalize to [0, 1]
            boundary_tensor = torch.clamp(boundary_tensor, 0, 1)
            
            # Convert to numpy (H, W)
            boundary_map = boundary_tensor.squeeze().cpu().numpy()
        
        return boundary_map
    
    def save_boundary(self, boundary_map, output_path):
        """
        Save boundary map as image
        
        Args:
            boundary_map: numpy array (H, W) with values in [0, 1]
            output_path: Path to save boundary image
        """
        # Convert to [0, 255]
        boundary_uint8 = (boundary_map * 255).astype(np.uint8)
        
        # Save
        cv2.imwrite(output_path, boundary_uint8)
    
    def process_single_mask(self, mask_path, output_path):
        """
        Process single mask: extract boundary and save
        
        Args:
            mask_path: Path to input mask
            output_path: Path to save boundary map
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load mask
            mask_tensor = self.load_mask(mask_path)
            
            # Extract boundary
            boundary_map = self.extract_boundary(mask_tensor)
            
            # Save boundary
            self.save_boundary(boundary_map, output_path)
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to process {mask_path}: {str(e)}")
            return False
    
    def process_directory(self, mask_dir, output_dir, file_extension='.png'):
        """
        Process all masks in directory
        
        Args:
            mask_dir: Directory containing mask images
            output_dir: Directory to save boundary maps
            file_extension: Image file extension (default: .png)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Output directory: {output_dir}")
        
        # Get all mask files
        mask_dir = Path(mask_dir)
        mask_files = list(mask_dir.glob(f'*{file_extension}'))
        
        if not mask_files:
            print(f"[WARNING] No mask files found in {mask_dir}")
            return
        
        print(f"[INFO] Found {len(mask_files)} mask files")
        
        # Process each mask
        success_count = 0
        fail_count = 0
        
        for mask_path in tqdm(mask_files, desc="Extracting boundaries"):
            # Get output path (same filename)
            output_path = os.path.join(output_dir, mask_path.name)
            
            # Process
            if self.process_single_mask(str(mask_path), output_path):
                success_count += 1
            else:
                fail_count += 1
        
        # Summary
        print(f"\n[SUMMARY]")
        print(f"  Success: {success_count}/{len(mask_files)}")
        print(f"  Failed:  {fail_count}/{len(mask_files)}")
        print(f"  Saved to: {output_dir}")


def main():
    """Main function"""
    
    # Configuration
    TRAIN_ROOT = "../MHCD_seg/train"  # ← Change this to your train directory
    MASK_SUBDIR = "masks"  # Masks are in train/masks/
    BOUNDARY_SUBDIR = "boundaries"  # Save boundaries to train/boundaries/
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    mask_dir = os.path.join(TRAIN_ROOT, MASK_SUBDIR)
    boundary_dir = os.path.join(TRAIN_ROOT, BOUNDARY_SUBDIR)
    
    # Verify mask directory exists
    if not os.path.exists(mask_dir):
        print(f"[ERROR] Mask directory not found: {mask_dir}")
        return
    
    print("="*80)
    print("BOUNDARY MAP EXTRACTION")
    print("="*80)
    print(f"Mask directory:      {mask_dir}")
    print(f"Boundary directory:  {boundary_dir}")
    print(f"Device:              {DEVICE}")
    print("="*80)
    
    # Create extractor
    extractor = BoundaryExtractor(device=DEVICE)
    
    # Process all masks
    extractor.process_directory(mask_dir, boundary_dir)
    
    print("="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()