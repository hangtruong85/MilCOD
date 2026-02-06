import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.unet3plus import UNet3Plus_B3_BEM


def load_model(weight_path, device="cuda"):
    model = UNet3Plus_B3_BEM().to(device)
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def build_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])


@torch.no_grad()
def infer_folder(
    model,
    input_folder,
    save_folder="infer_out",
    img_size=256,
    device="cuda",
    exts=(".jpg", ".png", ".jpeg", ".bmp"),
):
    os.makedirs(save_folder, exist_ok=True)

    transform = build_transform(img_size)

    # Collect list images
    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(exts)
    ]
    if len(files) == 0:
        print("Không tìm thấy ảnh hợp lệ trong folder:", input_folder)
        return

    print(f"===> Bắt đầu inference {len(files)} ảnh")
    t0 = time.time()

    for path in tqdm(files, desc="Infer"):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Bỏ qua ảnh lỗi: {path}")
            continue

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocess
        t = transform(image=img2)
        tensor = t["image"].unsqueeze(0).to(device)

        # forward
        mask_logits, _ = model(tensor)
        prob = torch.sigmoid(mask_logits)[0,0].cpu().numpy()

        # convert mask
        mask255 = (prob*255).astype(np.uint8)

        ### restore original size
        h, w = img.shape[:2]
        mask_resized = cv2.resize(mask255, (w,h), interpolation=cv2.INTER_NEAREST)

        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(save_folder, f"{name}_mask.png")
        cv2.imwrite(out_path, mask_resized)

    print(f"===> Done. Masks saved → {save_folder}")
    print(f"===> Total time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--out", type=str, default="infer_out")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    infer_folder(
        model,
        args.input_folder,
        save_folder=args.out,
        img_size=args.img_size,
        device=args.device,
    )

