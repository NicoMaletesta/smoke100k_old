#!/usr/bin/env python3
"""
eval_smoke.py
Inference-only script for Smoke segmentation (NO ground-truth required).
Supports:
  - image: predict single image, save binary mask and optional overlay
  - video: predict frames and save video with prediction contours
  - batch-folder: sample N images from a folder (pseudo-random) and run batch inference

Usage examples:
  python infer_smoke.py --mode image --model_path /path/to/ckpt.pth --image_path img.png --out_dir ./out --size 512
  python infer_smoke.py --mode video --model_path /path/to/ckpt.pth --video_path vid.mp4 --out_dir ./out --size 512
  python infer_smoke.py --mode folder --batch --image_dir ./images --out_dir ./out --batch_size 8 --sample_n 100 --seed 42
"""

import os
from pathlib import Path
import argparse
import time
import random

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.model import UNetSegmenter
from src.utils import load_checkpoint

# ------------------ model/load utilities (unchanged) ------------------
def load_model(path: str, device: torch.device):
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    model = UNetSegmenter(out_channels=1, pretrained=False).to(device)
    try:
        load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=device)
    except Exception:
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

# ------------------ preprocessing / helpers (unchanged) ------------------
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

def save_mask(mask_np, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image = __import__('PIL').Image
    Image.fromarray((mask_np*255).astype('uint8')).save(out_path)

def overlay_rgb(orig_rgb: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.6):
    out = orig_rgb.copy().astype(np.float32)
    red = np.array([255,0,0], dtype=np.float32)
    if pred_mask.any():
        out[pred_mask.astype(bool)] = (1-alpha)*out[pred_mask.astype(bool)] + alpha*red
    return out.astype('uint8')

# ------------------ single-image mode (unchanged) ------------------
def mode_image(model, image_path, size, device, out_dir, thr=0.5, save_overlay=True):
    os.makedirs(out_dir, exist_ok=True)
    inp, orig = preprocess_image(image_path, size, device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)
    pred_mask = probs_to_mask(probs, thr=thr)
    stem = Path(image_path).stem
    preds_dir = Path(out_dir)/'predictions'
    preds_dir.mkdir(parents=True, exist_ok=True)
    pred_path = preds_dir / f"{stem}_pred.png"
    save_mask(pred_mask, str(pred_path))
    print(f"✅ Saved prediction: {pred_path}")
    if save_overlay:
        overlays = Path(out_dir)/'overlays'
        overlays.mkdir(parents=True, exist_ok=True)
        h0, w0 = orig.shape[:2]
        pred_full = cv2.resize((pred_mask * 255).astype('uint8'), (w0, h0), interpolation=cv2.INTER_NEAREST) > 127
        def _overlay_rgb(orig_rgb: np.ndarray, pred_mask_bool: np.ndarray, alpha: float = 0.6) -> np.ndarray:
            out = orig_rgb.copy().astype(np.float32)
            red = np.array([255, 0, 0], dtype=np.float32)
            if pred_mask_bool.shape != out.shape[:2]:
                pred_mask_bool = cv2.resize((pred_mask_bool.astype('uint8') * 255), (out.shape[1], out.shape[0]),
                                            interpolation=cv2.INTER_NEAREST) > 127
            if pred_mask_bool.any():
                out[pred_mask_bool] = (1 - alpha) * out[pred_mask_bool] + alpha * red
            return out.astype('uint8')
        overlay_img = _overlay_rgb(orig, pred_full)
        ov_path = overlays / f"{stem}_overlay.png"
        cv2.imwrite(str(ov_path), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

# ------------------ video mode (unchanged) ------------------
def mode_video(model, video_path, size, device, out_dir, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    basename = Path(video_path).stem
    out_file = os.path.join(out_dir, f"{basename}_pred_contours.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_file, fourcc, fps, (w,h))
    pbar = tqdm(total=total, desc=f"Video {basename}", unit='frame')

    with torch.no_grad():
        frame_idx = 0
        start = time.time()
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rs = cv2.resize(rgb, (size,size))
            t = torch.from_numpy(rs.transpose(2,0,1)).unsqueeze(0).float().div(255.0).to(device)
            logits = model(t)
            probs = torch.sigmoid(logits)
            pred_mask = probs_to_mask(probs, thr=thr).astype('uint8')*255
            mask_o = cv2.resize(pred_mask, (w,h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out_frame = frame_bgr.copy()
            cv2.drawContours(out_frame, contours, -1, (0,0,255), 2)
            writer.write(out_frame)
            frame_idx += 1
            pbar.update(1)
    cap.release(); writer.release(); pbar.close()
    total_time = time.time() - start
    print(f"\n✅ Video saved -> {out_file}")
    print(f"frames: {frame_idx}/{total} time: {total_time:.1f}s avg FPS: {frame_idx/total_time:.1f}")

# ------------------ NEW: batch-folder mode ------------------
def mode_batch_folder(model, image_dir, out_dir, size, device, batch_size=8, thr=0.5,
                      sample_n=None, seed=42, save_probmaps=True, save_overlay=False, preprocess_fn=None):
    """
    Batch inference on a folder of images. Optionally sample `sample_n` images (pseudo-random, seedable).
    Saves:
      - out_dir/predictions/<stem>_pred.png
      - out_dir/probmaps/<stem>_prob.npy  (if save_probmaps=True)
      - out_dir/overlays/<stem>_overlay.png  (if save_overlay=True)
    """
    model.to(device)
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    preds_dir = Path(out_dir) / "predictions"
    preds_dir.mkdir(exist_ok=True)
    probs_dir = Path(out_dir) / "probmaps"
    if save_probmaps:
        probs_dir.mkdir(exist_ok=True)
    overlays_dir = Path(out_dir) / "overlays"
    if save_overlay:
        overlays_dir.mkdir(exist_ok=True)

    all_paths = sorted([p for p in Path(image_dir).glob("*") if p.suffix.lower() in ('.jpg','.png','.jpeg')])
    if sample_n is not None and sample_n < len(all_paths):
        random.seed(seed)
        sampled = random.sample(all_paths, sample_n)
        paths = sorted(sampled)
    else:
        paths = all_paths

    n = len(paths)
    if n == 0:
        print("[WARN] no images found in", image_dir)
        return

    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Batch infer"):
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
                    # create RGB overlay (red for pred)
                    overlay_img = overlay_rgb(orig_rgbs[j], bin_mask.astype(bool), alpha=0.6)
                    cv2.imwrite(str(overlays_dir / (base + "_overlay.png")), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

                # quick logging stats
                meanp = float(prob_resized.mean())
                p90 = float(np.percentile(prob_resized, 90))
                print(f"[INF] {p.name} mean_prob={meanp:.4f} p90={p90:.4f}")

    print("Batch inference completed. Results in:", out_dir)

# ------------------ CLI ------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Inference-only script for Smoke segmentation (image/video/folder)')
    p.add_argument('--mode', choices=['image','video','folder'], default="folder")
    p.add_argument('--model_path', type=str, default=r"/home/nicola/Scrivania/segmentation models/smoke/smoke_best.pth")
    p.add_argument('--image_path', type=str, default=r"/home/nicola/Scrivania/test image from the net/20680295374_7af01a40b6_o.jpg")
    p.add_argument('--video_path', type=str, default=r"/home/nicola/Scrivania/test image from the net/incendiosc.mp4")
    p.add_argument('--image_dir', type=str, default=r"/home/nicola/Scrivania/test image from the net")
    p.add_argument('--out_dir', type=str, default=r"/home/nicola/Scaricati/checkpointssept")
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--no_overlay', action='store_true', help='Do not save overlay for image mode')
    # batch options
    p.add_argument('--batch', action='store_true', help='Use batch-folder inference mode')
    p.add_argument('--batch_size', type=int, default=8, help='Batch size for folder inference')
    p.add_argument('--sample_n', type=int, default=10, help='If set, pseudo-randomly sample N images from folder')
    p.add_argument('--seed', type=int, default=42, help='Random seed for sampling images')
    p.add_argument('--save_probmaps', default = True, help='Save probability maps (.npy) for each image')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)

    if args.mode == 'image':
        assert args.image_path, '--image_path required for image mode'
        mode_image(model, args.image_path, args.size, device, args.out_dir, thr=args.threshold, save_overlay=not args.no_overlay)
    elif args.mode == 'video':
        assert args.video_path, '--video_path required for video mode'
        mode_video(model, args.video_path, args.size, device, args.out_dir, thr=args.threshold)
    else:
        assert args.image_dir, '--image_dir required for folder mode'
        if args.batch:
            mode_batch_folder(model, args.image_dir, args.out_dir, args.size, device,
                              batch_size=args.batch_size, thr=args.threshold,
                              sample_n=args.sample_n, seed=args.seed,
                              save_probmaps=args.save_probmaps, save_overlay=not args.no_overlay)
        else:
            # existing single-image loop (iterate images one by one)
            image_paths = sorted([p for p in Path(args.image_dir).glob('*') if p.suffix.lower() in ('.jpg','.png','.jpeg')])
            for p in image_paths:
                mode_image(model, str(p), args.size, device, args.out_dir, thr=args.threshold, save_overlay=not args.no_overlay)
