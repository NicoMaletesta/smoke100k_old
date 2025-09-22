#!/usr/bin/env python3
# src/debug_attention.py
"""
Script di debug per verificare che SEBlock e CBAM siano effettivamente usati
nel modello e per visualizzare le loro uscite/attivazioni.

Uso:
    python src/debug_attention.py --img /path/to/img.jpg --checkpoint /path/to/ckpt.pt --outdir ./attn_debug

Se checkpoint non fornito, usa modello istanziato (pretrained backbone True/False passato via flag).
"""
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# importa il modello e i moduli di attention (assume che src sia nel PYTHONPATH)
from src.model import UNetSegmenter
from src.attention import SEBlock, CBAM


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(path, size=(512,512)):
    img = Image.open(path).convert("RGB")
    img_resized = img.resize(size, resample=Image.BILINEAR)
    return img, img_resized


def preprocess(img_pil, size=(512,512), device='cpu'):
    tf = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    t = tf(img_pil).unsqueeze(0).to(device)
    return t


def save_channel_bar(weights, outpath, title="SE channel weights"):
    w = weights.copy()
    plt.figure(figsize=(10,3))
    plt.bar(np.arange(len(w)), w)
    plt.xlabel("Channel")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_spatial_overlay_pil(rgb_pil, spatial_map_2d, outpath, alpha=0.4, cmap='jet'):
    """
    Robust PIL-based overlay:
      - spatial_map_2d: numpy 2D array float in [0,1] (any size)
      - rgb_pil: PIL.Image RGB original image
      - produces PNG at same size as rgb_pil
    """
    # normalize spatial_map_2d to 0..1
    sm = spatial_map_2d.astype(np.float32)
    sm -= sm.min()
    if sm.max() > 0:
        sm = sm / (sm.max() + 1e-12)
    else:
        sm = np.zeros_like(sm, dtype=np.float32)

    # resize sm to original image size (PIL expects (width,height))
    target_size = (rgb_pil.size[0], rgb_pil.size[1])  # (W, H)
    sm_img = Image.fromarray((sm * 255).astype('uint8')).resize(target_size, resample=Image.BILINEAR)

    # apply colormap (matplotlib colormap returns RGBA 0..1)
    cmap_obj = matplotlib.colormaps[cmap]
    sm_norm = np.asarray(sm_img).astype(np.float32) / 255.0
    sm_col = cmap_obj(sm_norm)  # HxWx4 float
    sm_col = (sm_col * 255).astype('uint8')
    sm_col_pil = Image.fromarray(sm_col).convert("RGBA")  # RGBA

    # original as RGBA
    rgb_rgba = rgb_pil.convert("RGBA")

    # blend: create transparent layer and alpha_composite
    blended = Image.alpha_composite(rgb_rgba, Image.blend(Image.new("RGBA", target_size, (0,0,0,0)), sm_col_pil, alpha=alpha))

    # save
    blended.convert("RGB").save(outpath, quality=95)


def normalize_map(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    m -= m.min()
    if m.max() > 0:
        m /= (m.max() + 1e-12)
    return m


def tensor_to_numpy_map(t: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor (1,1,H,W) or (H,W) into a numpy 2D array.
    """
    if isinstance(t, torch.Tensor):
        t_cpu = t.detach().cpu()
        if t_cpu.ndim == 4:
            arr = t_cpu[0, 0].numpy()
        elif t_cpu.ndim == 3:
            # shape could be (1,H,W) or (H, W, 1). Try common cases:
            if t_cpu.shape[0] == 1:
                arr = t_cpu[0].numpy()
            else:
                arr = t_cpu.numpy().squeeze()
        elif t_cpu.ndim == 2:
            arr = t_cpu.numpy()
        else:
            arr = t_cpu.numpy().squeeze()
    else:
        arr = np.array(t)
        arr = np.squeeze(arr)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=r"/home/nicola/Scrivania/test image from the net/7.jpeg", help="Percorso immagine RGB")
    parser.add_argument("--checkpoint", default=None, help="Percorso checkpoint (opzionale)")
    parser.add_argument("--outdir", default=r"/home/nicola/Scaricati/checkpointssept22/attn_debug", help="Cartella output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-size", type=int, default=512, help="Size H=W per il modello")
    parser.add_argument("--pretrained-backbone", default=True, help="Se usare weights ImageNet per backbone se no checkpoint")
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha per l'overlay")
    parser.add_argument("--cmap", type=str, default="inferno", help="Colormap per l'overlay (matplotlib)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    # instantiate model (out_channels 1 o 2 a seconda del tuo progetto; qui non importante)
    model = UNetSegmenter(out_channels=1, pretrained=args.pretrained_backbone)
    model = model.to(device)
    model.eval()

    # optionally load checkpoint (solo state_dict)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        # strip 'module.' prefix if present (common)
        new_sd = {}
        for k, v in sd.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
        print(f"[INFO] Loaded checkpoint {args.checkpoint} (strict=False)")

    # load image and preprocess
    rgb_orig, rgb_resized = load_image(args.img, size=(args.input_size, args.input_size))
    x = preprocess(rgb_resized, size=(args.input_size, args.input_size), device=device)

    # forward
    with torch.no_grad():
        out = model(x)

    # try to find SE and CBAM buffers
    se_weights = None
    cbam_map = None

    # Try common attribute names used in the provided model implementation
    if hasattr(model, "attn_se") and hasattr(model.attn_se, "last_weights"):
        se_weights = model.attn_se.last_weights.cpu().numpy()
        print("[DEBUG] Found SEBlock.last_weights with shape", se_weights.shape)
    else:
        # try scanning modules for SEBlock
        for m in model.modules():
            if isinstance(m, SEBlock) and hasattr(m, "last_weights"):
                se_weights = m.last_weights.cpu().numpy()
                print("[DEBUG] Found SEBlock instance in model.modules()")
                break

    if hasattr(model, "attn_cbam") and hasattr(model.attn_cbam, "last_spatial"):
        cbam_map = model.attn_cbam.last_spatial  # keep as tensor for best quality
        print("[DEBUG] Found CBAM.last_spatial (tensor) with shape", tuple(cbam_map.shape))
    else:
        for m in model.modules():
            if isinstance(m, CBAM) and hasattr(m, "last_spatial"):
                cbam_map = m.last_spatial
                print("[DEBUG] Found CBAM instance in model.modules()")
                break

    # Save / visualize SE channel weights
    if se_weights is not None:
        se_w = se_weights.reshape(-1)
        np.save(os.path.join(args.outdir, "se_channel_weights.npy"), se_w)
        save_channel_bar(se_w, os.path.join(args.outdir, "se_channel_weights.png"), title="SE channel weights")
        print(f"[OUTPUT] SE channel weights saved to {args.outdir}")
    else:
        print("[WARN] SE channel weights not found in model. Ensure SEBlock registers `last_weights`.")

    # Save / visualize CBAM spatial map
    if cbam_map is not None:
        # convert to tensor on CPU
        if isinstance(cbam_map, np.ndarray):
            cb_t = torch.from_numpy(cbam_map).float()
        else:
            cb_t = cbam_map.detach().cpu().float()

        # Normalize and inspect raw small map
        raw_small = tensor_to_numpy_map(cb_t)
        raw_small_norm = normalize_map(raw_small.copy())
        print("[DEBUG] CBAM raw small map shape (HxW):", raw_small.shape,
              "min/max/mean/std:", float(raw_small.min()), float(raw_small.max()), float(raw_small.mean()), float(raw_small.std()))

        # Save raw small normalized
        np.save(os.path.join(args.outdir, "cbam_spatial_map_small.npy"), raw_small_norm)

        # Upsample using bilinear interpolation to original image resolution (use torch for smoothness)
        # target size for interpolate is (H, W)
        target_h, target_w = rgb_orig.size[1], rgb_orig.size[0]
        # ensure shape is [1,1,H,W]
        if cb_t.ndim == 4:
            cb_t4 = cb_t
        elif cb_t.ndim == 3:
            # could be (1,H,W) or (H,1,W) - try to shape to (1,1,H,W)
            if cb_t.shape[0] == 1:
                cb_t4 = cb_t.unsqueeze(1) if cb_t.ndim == 3 else cb_t.unsqueeze(0).unsqueeze(0)
            else:
                cb_t4 = cb_t.unsqueeze(0).unsqueeze(0)
        elif cb_t.ndim == 2:
            cb_t4 = cb_t.unsqueeze(0).unsqueeze(0)
        else:
            cb_t4 = cb_t.reshape(1, 1, *cb_t.shape[-2:])

        try:
            cb_up = F.interpolate(cb_t4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        except Exception as e:
            # fallback to numpy/PIL resize if interpolation fails
            print("[WARN] torch.interpolate failed:", e)
            cb_np_small = raw_small_norm
            cb_up_pil = Image.fromarray((cb_np_small * 255).astype(np.uint8)).resize((target_w, target_h), resample=Image.BILINEAR)
            cb_up = torch.from_numpy(np.asarray(cb_up_pil).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        sp = cb_up[0, 0].cpu().numpy()
        spn = normalize_map(sp)
        # Save upsampled normalized map
        np.save(os.path.join(args.outdir, "cbam_spatial_map_upsampled.npy"), spn)
        print("[OUTPUT] CBAM upsampled map shape:", spn.shape, "min/max/mean/std:", float(spn.min()), float(spn.max()), float(spn.mean()), float(spn.std()))

        # Blend overlay with PIL-based method (robust)
        overlay_path = os.path.join(args.outdir, "cbam_spatial_overlay.png")
        save_spatial_overlay_pil(rgb_orig, spn, overlay_path, alpha=args.alpha, cmap=args.cmap)
        print(f"[OUTPUT] CBAM spatial overlay (upsampled) saved to {overlay_path}")

    else:
        print("[WARN] CBAM spatial map not found in model. Ensure CBAM registers `last_spatial`.")

    print("[DONE] Debug outputs in", args.outdir)


if __name__ == "__main__":
    main()
