import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MHCDDatasetWithDepth(Dataset):
    """
    MHCD Dataset with RGB + Depth + Mask support
    Uses ReplayCompose to ensure synchronized augmentation
    
    Directory structure:
        root/
        ├── train/
        │   ├── images/  # RGB images (.jpg)
        │   ├── depth/   # Depth maps (.png)
        │   └── masks/   # Segmentation masks (.png)
        └── val/
            ├── images/
            ├── depth/
            └── masks/
    
    Returns:
        rgb: (3, H, W) - Normalized with ImageNet mean/std
        depth: (1, H, W) - Normalized to [0, 1]
        mask: (1, H, W) - Binary {0, 1}
    """
    
    def __init__(self, root, split="train", img_size=256, augment=True, 
                 use_depth=True, logger=None):
        """
        Args:
            root: Root directory of dataset
            split: 'train' or 'val'
            img_size: Target image size
            augment: Whether to apply augmentation (only for train)
            use_depth: Whether to load depth maps
            logger: Optional logger
        """
        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.depth_dir = os.path.join(root, split, "depth")
        
        self.use_depth = use_depth
        self.img_size = img_size
        self.augment = augment and (split == "train")
        
        # Get image paths
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        assert len(self.img_paths) > 0, f"No .jpg found in {self.img_dir}"
        
        # Build mask paths
        self.mask_paths = []
        for ip in self.img_paths:
            basename = os.path.splitext(os.path.basename(ip))[0]
            mask_path = os.path.join(self.mask_dir, basename + ".png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            self.mask_paths.append(mask_path)
        
        # Build depth paths
        if self.use_depth:
            if not os.path.exists(self.depth_dir):
                if logger:
                    logger.warning(f"Depth directory not found: {self.depth_dir}")
                    logger.warning("Depth loading disabled!")
                self.use_depth = False
            else:
                self.depth_paths = []
                for ip in self.img_paths:
                    basename = os.path.splitext(os.path.basename(ip))[0]
                    depth_path = os.path.join(self.depth_dir, basename + ".png")
                    if not os.path.exists(depth_path):
                        raise FileNotFoundError(f"Depth not found: {depth_path}")
                    self.depth_paths.append(depth_path)
        
        # Setup transforms using ReplayCompose for synchronization
        if self.augment:
            # Geometric transforms (synchronized across RGB, depth, mask)
            self.geometric_aug = A.ReplayCompose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.4
                ),
            ])
            
            # Color augmentation (RGB only)
            self.color_aug = A.Compose([
                A.RandomBrightnessContrast(p=0.2),
            ])
        else:
            # Validation: only resize
            self.geometric_aug = A.ReplayCompose([
                A.Resize(img_size, img_size),
            ])
            self.color_aug = None
        
        # Normalization for RGB
        self.rgb_normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
        
        # Logging
        log_msg = f"[Dataset] {split}: images={len(self.img_paths)}, masks={len(self.mask_paths)}"
        if self.use_depth:
            log_msg += f", depth={len(self.depth_paths)}"
        else:
            log_msg += ", depth=DISABLED"
        
        print(log_msg)
        if logger:
            logger.info(log_msg)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        Get a training sample
        
        Returns:
            rgb: RGB image tensor (3, H, W) - normalized
            depth: Depth map tensor (1, H, W) - [0, 1]
            mask: Segmentation mask (1, H, W) - binary
        """
        # Load RGB image
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask (keep original 0-255 range for augmentation)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Load depth
        if self.use_depth:
            depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_GRAYSCALE)
        else:
            # Fallback: pseudo depth from RGB grayscale
            depth = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply geometric augmentation to RGB and record transforms
        geometric_result = self.geometric_aug(image=img)
        img = geometric_result['image']
        
        # Apply SAME geometric transforms to depth and mask using replay
        depth = A.ReplayCompose.replay(geometric_result['replay'], image=depth)['image']
        mask = A.ReplayCompose.replay(geometric_result['replay'], image=mask)['image']
        
        # Apply color augmentation to RGB only
        if self.color_aug is not None:
            img = self.color_aug(image=img)['image']
        
        # Normalize RGB
        img = self.rgb_normalize(image=img)['image']
        
        # Normalize depth to [0, 1]
        depth = depth.astype("float32") / 255.0
        
        # Binarize mask (mask is in 0-255 range after resize)
        mask = (mask > 127).astype("float32")
        
        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        return img, depth, mask

class MHCDDataset(Dataset):
    def __init__(self, root, split="train", img_size=256, augment=True, logger=None):
        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        assert len(self.img_paths) > 0, f"No .jpg found in {self.img_dir}"

        self.mask_paths = []
        for ip in self.img_paths:
            b = os.path.splitext(os.path.basename(ip))[0]
            mp = os.path.join(self.mask_dir, b + ".png")
            if not os.path.exists(mp):
                raise FileNotFoundError(mp)
            self.mask_paths.append(mp)

        self.img_size = img_size

        if augment and split == "train":
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.4
                ),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])

        print(f"[Dataset] {split}: images={len(self.img_paths)}, masks={len(self.mask_paths)}")
        if logger:
            logger.info(f"[Dataset] {split}: images={len(self.img_paths)}, masks={len(self.mask_paths)}")
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")

        sample = self.transform(image=img, mask=mask)
        img = sample["image"]
        mask = sample["mask"].unsqueeze(0)

        return img, mask
