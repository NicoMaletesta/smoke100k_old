
# ===== file: debug_train.py =====
#!/usr/bin/env python3
"""
debug_train.py
Utility visuale per controllare che il training / dataset / checkpoint siano coerenti.
- carica un modello (opzionale)
- prende N immagini dal val set
- calcola predizioni (probabilities)
- salva per ciascuna immagine: input, GT heatmap, pred probability heatmap, pred binary mask (usando binary_threshold), overlay (image + pred heatmap), overlay_binary

Usalo per sanity-check prima di lanciare run lunghi.
"""
import os
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.dataset import SmokeDataset, overlay_mask_on_image, get_spatial_transforms
from src.model import UNetSegmenter
from config import TRAIN


def save_png(img_arr, path):
    Image.fromarray(img_arr).save(path)


def prob_to_uint8(prob_arr):
    return (np.clip(prob_arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def run_debug(checkpoint: str, n: int, out_dir: str, target_mode: str, binary_threshold: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSegmenter(out_channels=1, pretrained=True).to(device)
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint {checkpoint} ...")
        ckpt = torch.load(checkpoint, map_location=device, weights_only = True)
        # try common keys
        if isinstance(ckpt, dict):
            for k in ("model_state", "state_dict", "model", "net"):
                if k in ckpt:
                    try:
                        model.load_state_dict(ckpt[k])
                        break
                    except Exception:
                        pass
            else:
                try:
                    model.load_state_dict(ckpt)
                except Exception:
                    print("Warning: could not load model weights from checkpoint cleanly; proceeding with random init.")
        else:
            try:
                model.load_state_dict(ckpt)
            except Exception:
                print("Warning: could not load model weights from checkpoint cleanly; proceeding with random init.")
    else:
        print("No valid checkpoint provided — model will be random initialized (useful for IO/debug only).")

    model.eval()

    # build val dataset
    base = TRAIN["data_dir"]
    val_img_dir = os.path.join(base, "val", "images")
    val_mask_dir = os.path.join(base, "val", "masks")

    ds = SmokeDataset(images_dir=val_img_dir, masks_dir=val_mask_dir, color_transforms=None,
                      spatial_transforms_factory=None, target_mode=target_mode, binary_threshold=binary_threshold, preload=False)

    ensure_dir(out_dir)

    # take n random samples
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    for i, idx in enumerate(idxs):
        sample = ds[int(idx)]
        img_t = sample["image"]  # [3,H,W]
        gt_t = sample["mask"]    # [1,H,W]
        meta = sample.get("meta", {})

        img_np = (img_t.permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
        gt_np = gt_t.squeeze(0).cpu().numpy()

        # forward
        with torch.no_grad():
            inp = img_t.unsqueeze(0).to(device)
            logits = model(inp)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

        # binary pred
        bin_pred = (probs > binary_threshold).astype(np.uint8)

        # save components
        base_name = f"debug_{i:03d}"
        save_png(img_np, os.path.join(out_dir, base_name + "_image.png"))
        save_png(prob_to_uint8(probs), os.path.join(out_dir, base_name + "_pred_prob.png"))
        save_png(prob_to_uint8(gt_np), os.path.join(out_dir, base_name + "_gt_heat.png"))
        save_png((bin_pred * 255).astype(np.uint8), os.path.join(out_dir, base_name + "_pred_bin.png"))

        # overlays (use dataset utility)
        img_pil = Image.fromarray(img_np)
        overlay_pred = overlay_mask_on_image(img_pil, probs, alpha=0.45)
        overlay_gt = overlay_mask_on_image(img_pil, gt_np, alpha=0.45)
        overlay_bin = overlay_mask_on_image(img_pil, bin_pred.astype(float), alpha=0.45)

        overlay_pred.save(os.path.join(out_dir, base_name + "_overlay_pred.png"))
        overlay_gt.save(os.path.join(out_dir, base_name + "_overlay_gt.png"))
        overlay_bin.save(os.path.join(out_dir, base_name + "_overlay_pred_bin.png"))

        # meta
        with open(os.path.join(out_dir, base_name + "_meta.txt"), "w") as fh:
            fh.write(str(meta))

    print(f"Saved {len(idxs)} debug samples to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug visual checks for dataset/model/checkpoint")
    parser.add_argument("--checkpoint", type=str, default=TRAIN.get("resume", None))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default=r"/home/nicola/Scaricati/checkpointssept/debug")
    parser.add_argument("--target_mode", type=str, choices=["soft","binary"], default=TRAIN.get("target_mode", "soft"))
    parser.add_argument("--binary_threshold", type=float, default=TRAIN.get("binary_threshold", 0.5))
    args = parser.parse_args()

    run_debug(args.checkpoint, args.n, args.out_dir, args.target_mode, args.binary_threshold)
