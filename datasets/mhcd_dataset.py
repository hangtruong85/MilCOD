import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
