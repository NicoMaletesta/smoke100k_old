#!/usr/bin/env python3
"""
check_mask.py

Script di utilità per verificare che una coppia (immagine RGB, mask GT) sia processata
correttamente dallo stesso pipeline del tuo dataset.py.

Output:
 - stampa diagnostica (shape, dtype, min/max, valori unici in modalità binaria)
 - salva overlay opacizzati della GT sull'originale (soft e, se richiesto, binario)

Usage:
    python3 check_mask.py --image path/to/img.png --mask path/to/mask.png --outdir ./check_out

Lo script prova prima a riusare le funzioni da dataset.py (se si trova nello stesso folder).
Se non trova dataset.py, usa implementazioni locali equivalenti.
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Try to import helpers from dataset.py if available
try:
    from dataset import pil_loader, load_mask_as_float, overlay_mask_on_image
    print("[INFO] Imported helpers from dataset.py")
except Exception:
    print("[INFO] dataset.py non trovato o import fallito: uso helper locali")

    def pil_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def load_mask_as_float(mask_path: str) -> np.ndarray:
        im = Image.open(mask_path)
        arr = np.array(im)
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        maxv = float(arr.max()) if arr.size > 0 else 1.0
        if maxv > 255:
            return arr.astype(np.float32) / 65535.0
        denom = maxv if maxv > 0 else 1.0
        return arr.astype(np.float32) / denom

    # simple overlay using matplotlib-like colormap (magma) equivalent via PIL
    try:
        import matplotlib as mpl
        def overlay_mask_on_image(img: Image.Image, mask_arr: np.ndarray, alpha: float = 0.45) -> Image.Image:
            h, w = mask_arr.shape
            img = img.convert("RGBA").resize((w, h))
            cmap = mpl.colormaps.get("magma")
            colored = (cmap(mask_arr)[:, :, :3] * 255).astype(np.uint8)
            heat = Image.fromarray(colored).convert("RGBA")
            blended = Image.blend(img, heat, alpha=alpha)
            return blended
    except Exception:
        # fallback grayscale overlay (red mask)
        def overlay_mask_on_image(img: Image.Image, mask_arr: np.ndarray, alpha: float = 0.45) -> Image.Image:
            h, w = mask_arr.shape
            img = img.convert("RGBA").resize((w, h))
            mask_u8 = (np.clip(mask_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            red = Image.fromarray(mask_u8).convert("L")
            red_rgb = Image.merge("RGBA", [red, Image.new('L', red.size, 0), Image.new('L', red.size, 0), red])
            blended = Image.blend(img, red_rgb, alpha=alpha)
            return blended


def normalize_mask_for_overlay(mask):
    """Normalize mask to a 2D numpy array float32 in [0,1].

    Accepts:
      - numpy arrays (H,W), (H,W,3), or (1,H,W)
      - torch tensors with same shapes
    Returns:
      - numpy ndarray shape (H,W) dtype float32 in [0,1]

    Raises ValueError on unexpected shapes.
    """
    # convert torch -> numpy
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.array(mask)

    # channel-first single channel (1,H,W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    # channel-last RGB/RGBA (H,W,3) or (H,W,4)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = arr.mean(axis=2)

    if arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape for overlay: {arr.shape}")

    arr = arr.astype(np.float32)
    # normalize if in 0..255 or 0..65535
    maxv = float(arr.max()) if arr.size > 0 else 1.0
    if maxv > 1.01:
        if maxv > 65535 / 2:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def to_tensor_and_check(img_pil: Image.Image, mask_arr: np.ndarray, target_mode: str = "soft", binary_threshold: float = 0.5):
    """Return image_tensor [3,H,W], mask_tensor [1,H,W] (float32 in [0,1]) and some diagnostics."""
    # ensure sizes
    img_w, img_h = img_pil.size
    if mask_arr.ndim == 3:
        # if RGB mask, convert to grayscale via mean (channel-last)
        mask_arr = mask_arr.mean(axis=2)
    mask_h, mask_w = mask_arr.shape
    if (img_w, img_h) != (mask_w, mask_h):
        # resize mask to image size using bilinear for soft, nearest for binary
        import PIL
        resample = PIL.Image.BILINEAR if target_mode == "soft" else PIL.Image.NEAREST
        mask_pil = Image.fromarray((np.clip(mask_arr,0,1)*255.0).astype(np.uint8)).resize((img_w, img_h), resample=resample)
        mask_arr = np.array(mask_pil).astype(np.float32) / 255.0

    image_tensor = TF.to_tensor(img_pil).type(torch.float32)
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).type(torch.float32)

    if target_mode == "binary":
        mask_tensor = (mask_tensor > binary_threshold).float()

    diag = {
        "image_shape": tuple(image_tensor.shape),
        "image_dtype": str(image_tensor.dtype),
        "image_min_max": (float(image_tensor.min()), float(image_tensor.max())),
        "mask_shape": tuple(mask_tensor.shape),
        "mask_dtype": str(mask_tensor.dtype),
        "mask_min_max": (float(mask_tensor.min()), float(mask_tensor.max())),
    }
    if target_mode == "binary":
        diag["mask_unique"] = sorted([int(x) for x in torch.unique(mask_tensor).cpu().numpy()])
    return image_tensor, mask_tensor, diag


def main():
    parser = argparse.ArgumentParser(description="Check processing of a single image+GT mask pair")
    parser.add_argument("--image", default= r"/home/nicola/Immagini/Video Frames - Regione Calabria/reg cal 26/images/cutout_regcal_26_046.png", help="Path to RGB image")
    parser.add_argument("--mask", default= r"/home/nicola/Immagini/Video Frames - Regione Calabria/reg cal 26/masks/cutout_regcal_26_046.png", help="Path to GT mask (any bitdepth) ")
    parser.add_argument("--outdir", default=r"/home/nicola/Immagini/maskcheck", help="Folder where overlays and diagnostics are written")
    parser.add_argument("--binary","-b", default= True, help="Also produce binary-processed outputs (threshold default 0.5)")
    parser.add_argument("--threshold","-t", type=float, default=0.1, help="Threshold for binary mode")
    parser.add_argument("--alpha", type=float, default=0.65, help="Overlay alpha for visualization")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img = pil_loader(args.image)
    mask_arr = load_mask_as_float(args.mask)

    # normalize mask for overlay (this fixes the ValueError if mask has shape (1,H,W) or (H,W,3))
    try:
        mask_for_overlay = normalize_mask_for_overlay(mask_arr)
    except Exception as e:
        print(f"[ERROR] normalize_mask_for_overlay failed: {e}")
        raise

    # soft processing
    img_t, mask_t_soft, diag_soft = to_tensor_and_check(img, mask_arr, target_mode="soft", binary_threshold=args.threshold)
    print("=== SOFT MODE ===")
    for k, v in diag_soft.items():
        print(f"{k}: {v}")

    # save overlay soft (use normalized 2D mask)
    overlay_soft = overlay_mask_on_image(img, mask_for_overlay, alpha=args.alpha)
    out_soft = os.path.join(args.outdir, "overlay_soft.png")
    overlay_soft.save(out_soft)
    print(f"Saved overlay (soft) -> {out_soft}")

    # binary processing
    img_t_b, mask_t_bin, diag_bin = to_tensor_and_check(img, mask_arr, target_mode="binary", binary_threshold=args.threshold)
    print("=== BINARY MODE ===")
    for k, v in diag_bin.items():
        print(f"{k}: {v}")

    # save binary mask and overlay
    bin_mask_u8 = (mask_t_bin.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(bin_mask_u8).save(os.path.join(args.outdir, "mask_binary.png"))

    # create overlay from binary mask (normalize first)
    mask_bin_for_overlay = normalize_mask_for_overlay(bin_mask_u8)
    overlay_bin = overlay_mask_on_image(img, mask_bin_for_overlay, alpha=args.alpha)
    out_bin = os.path.join(args.outdir, "overlay_binary.png")
    overlay_bin.save(out_bin)
    print(f"Saved mask (binary) -> {os.path.join(args.outdir, 'mask_binary.png')}")
    print(f"Saved overlay (binary) -> {out_bin}")

    # also save the original image copy for convenience
    img.save(os.path.join(args.outdir, "image_original.png"))

    # save a tiny diagnostics file
    diag_all = {
        "soft": diag_soft,
        "binary": diag_bin,
        "image": os.path.abspath(args.image),
        "mask": os.path.abspath(args.mask),
    }
    import json
    with open(os.path.join(args.outdir, "diagnostics.json"), "w") as fh:
        json.dump(diag_all, fh, indent=2)
    print(f"Saved diagnostics -> {os.path.join(args.outdir, 'diagnostics.json')}")


if __name__ == '__main__':
    main()

