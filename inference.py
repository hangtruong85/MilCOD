import os
import cv2
import torch
import numpy as np
from models.unet3plus import UNet3Plus_B3_BEM
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_model(weight_path, device="cuda"):
    model = UNet3Plus_B3_BEM().to(device)
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def preprocess_image(img_path, img_size=256):
    """Đọc ảnh + resize + normalize giống lúc train."""
    image = cv2.imread(img_path)
    assert image is not None, f"Không thể đọc ảnh: {img_path}"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    t = transform(image=image)
    return t["image"].unsqueeze(0)  # (1,3,H,W)


def inference_single(
    model,
    img_path,
    save_dir="infer_out",
    img_size=256,
    device="cuda",
):
    os.makedirs(save_dir, exist_ok=True)

    # ---- preprocess ----
    tensor = preprocess_image(img_path, img_size).to(device)

    # ---- forward ----
    with torch.no_grad():
        mask_logits, _ = model(tensor)
        mask_prob = torch.sigmoid(mask_logits)[0,0]  # (H,W)

    # ---- post-process ----
    mask = (mask_prob.cpu().numpy() * 255).astype(np.uint8)

    # ---- save ----
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(save_dir, base + "_mask.png")
    cv2.imwrite(out_path, mask)

    return out_path
