#!/usr/bin/env python3
"""
eval_models.py
Run batch-folder inference for multiple checkpoints.

For each .pth file in --models_dir this script will:
  - load the model weights
  - run folder-mode inference on --image_dir (using batch_size, size, threshold, ...)
  - save outputs under OUT_DIR/<ckpt_stem>/
    (predictions/, probmaps/, overlays/)

Usage example:
  python eval_models.py \
    --models_dir ./checkpoints/sweep \
    --image_dir ./data/processed/smoke/test/images \
    --out_dir ./infer_results \
    --size 512 --threshold 0.6 --batch_size 8 --save_probmaps
"""
import os
import time
import random
from pathlib import Path
import argparse
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from src.model import UNetSegmenter
from src.utils import load_checkpoint

# -------------------- helpers (copied / adapted from eval_smoke.py) --------------------
def load_model_from_ckpt(path: str, device: torch.device) -> torch.nn.Module:
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    model = UNetSegmenter(out_channels=1, pretrained=False).to(device)
    try:
        # try loading using your util (supports different payload formats)
        load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=device)
    except Exception:
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

def preprocess_image(path: str, size: int, device: torch.device):
    img = Image.open(path).convert('RGB')
    orig = np.array(img)
    img_r = img.resize((size, size), resample=Image.BILINEAR)
    t = torch.from_numpy(np.array(img_r).transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)
    return t, orig

def probs_to_mask(probs: torch.Tensor, thr: float = 0.5):
    p = probs.detach().cpu()
    if p.dim() == 4 and p.size(1) == 1:
        p = p.squeeze(0).squeeze(0)
    elif p.dim() == 3:
        p = p.squeeze(0)
    return (p.numpy() > thr).astype('uint8')

def overlay_rgb(orig_rgb: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.6):
    out = orig_rgb.copy().astype(np.float32)
    red = np.array([255,0,0], dtype=np.float32)
    if pred_mask.any():
        out[pred_mask.astype(bool)] = (1-alpha)*out[pred_mask.astype(bool)] + alpha*red
    return out.astype('uint8')

# -------------------- core batch runner --------------------
def run_folder_for_checkpoint(model: torch.nn.Module,
                              model_path: str,
                              image_dir: str,
                              out_subdir: str,
                              size: int = 512,
                              device: Optional[torch.device] = None,
                              batch_size: int = 8,
                              thr: float = 0.5,
                              sample_n: Optional[int] = None,
                              seed: int = 42,
                              save_probmaps: bool = False,
                              save_overlay: bool = False,
                              preprocess_fn = None):
    """
    Runs folder-mode inference using the provided already-loaded model.
    Results saved under out_subdir (predictions/, probmaps/, overlays/).
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device).eval()

    out_base = Path(out_subdir)
    preds_dir = out_base / "predictions"
    probs_dir = out_base / "probmaps"
    overlays_dir = out_base / "overlays"

    preds_dir.mkdir(parents=True, exist_ok=True)
    if save_probmaps:
        probs_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted([p for p in Path(image_dir).glob("*") if p.suffix.lower() in ('.jpg','.png','.jpeg')])
    if sample_n is not None and sample_n < len(all_paths):
        random.seed(seed)
        sampled = random.sample(all_paths, sample_n)
        paths = sorted(sampled)
    else:
        paths = all_paths

    n = len(paths)
    if n == 0:
        print(f"[WARN] No images found in {image_dir}")
        return

    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc=f"Infer {Path(model_path).stem}", unit="batch"):
            batch_paths = paths[i:i+batch_size]
            tensors = []
            orig_sizes = []
            orig_rgbs = []
            for p in batch_paths:
                pil = Image.open(p).convert('RGB')
                orig = np.array(pil)
                orig_sizes.append(orig.shape[:2])  # H,W
                orig_rgbs.append(orig)
                if preprocess_fn is not None:
                    t = preprocess_fn(str(p), size, device)  # must return 1,C,H,W
                else:
                    t, _ = preprocess_image(str(p), size, device)
                tensors.append(t)
            batch = torch.cat(tensors, dim=0)  # B,C,H,W

            # forward (use amp if cuda available)
            try:
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = model(batch)
            except Exception:
                logits = model(batch)
            probs = torch.sigmoid(logits)  # B,1,H',W'

            # to cpu numpy
            probs_cpu = probs.detach().cpu().numpy()  # B,1,Hp,Wp
            for j, p in enumerate(batch_paths):
                base = p.stem
                prob = probs_cpu[j,0]  # Hp x Wp
                h0, w0 = orig_sizes[j]
                # resize prob back to original size (cv2 resize expects (width,height))
                prob_resized = cv2.resize((prob * 255.0).astype('uint8'), (w0, h0), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
                if save_probmaps:
                    np.save(str(probs_dir / (base + "_prob.npy")), prob_resized)
                bin_mask = (prob_resized > thr).astype('uint8') * 255
                Image.fromarray(bin_mask).save(str(preds_dir / (base + "_pred.png")))

                if save_overlay:
                    overlay_img = overlay_rgb(orig_rgbs[j], bin_mask.astype(bool), alpha=0.6)
                    cv2.imwrite(str(overlays_dir / (base + "_overlay.png")), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

                # quick logging stats per image
                meanp = float(prob_resized.mean())
                p90 = float(np.percentile(prob_resized, 90))
                print(f"[INF] {p.name} mean_prob={meanp:.4f} p90={p90:.4f}")

    elapsed = time.time() - start_time
    print(f"[DONE] model={Path(model_path).name} images={n} time={elapsed:.1f}s avg FPS={n/elapsed:.2f}")
    print("Results saved in:", out_base)

# -------------------- main CLI --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run folder-mode inference for all .pth in a folder")
    parser.add_argument('--models_dir', type=str, default=r"/home/nicola/Documenti/smokedetector9sept", help='Directory containing .pth checkpoint files')
    parser.add_argument('--image_dir', type=str, default=r"/home/nicola/Scrivania/test image from the net", help='Directory with images to run inference on')
    parser.add_argument('--out_dir', type=str, default=r"/home/nicola/Documenti/smokedetector9sept/infer", help='Base output directory (per-checkpoint subfolders will be created)')
    parser.add_argument('--size', type=int, default=512, help='Resize size for model input (H=W)')
    parser.add_argument('--threshold', type=float, default=0.8, help='Binarization threshold for probability -> mask')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_n', type=int, default=5, help='If set, pseudo-randomly sample N images from folder')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_probmaps', action='store_true', help='Save probability maps (.npy)')
    parser.add_argument('--save_overlay', default= True, help='Save RGB overlays')
    parser.add_argument('--model_pattern', type=str, default="*.pth", help='Glob pattern to select model files (default *.pth)')
    parser.add_argument('--device', type=str, default=None, help='Device string (cuda or cpu). Default: auto-detect')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    models_dir = Path(args.models_dir)
    assert models_dir.exists() and models_dir.is_dir(), f"models_dir not found: {models_dir}"

    model_files = sorted(models_dir.glob(args.model_pattern))
    if len(model_files) == 0:
        print(f"[WARN] No model files found with pattern {args.model_pattern} in {models_dir}")
        exit(0)

    print(f"Found {len(model_files)} models. Device: {device}")
    for model_path in model_files:
        ckpt_name = model_path.stem
        out_subdir = Path(args.out_dir) / ckpt_name
        # skip if already exists (optional) - here we always run; you can toggle below
        # if out_subdir.exists():
        #     print(f"[SKIP] outputs for {ckpt_name} already exist at {out_subdir}; remove or rename to re-run.")
        #     continue

        print(f"\n=== Running inference for model: {model_path.name} -> {out_subdir} ===")
        try:
            model = load_model_from_ckpt(str(model_path), device)
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_path}: {e}")
            continue

        run_folder_for_checkpoint(model=model,
                                  model_path=str(model_path),
                                  image_dir=args.image_dir,
                                  out_subdir=str(out_subdir),
                                  size=args.size,
                                  device=device,
                                  batch_size=args.batch_size,
                                  thr=args.threshold,
                                  sample_n=args.sample_n,
                                  seed=args.seed,
                                  save_probmaps=args.save_probmaps,
                                  save_overlay=args.save_overlay)

    print("\nAll done.")
