#!/usr/bin/env python3
"""
infer_probability.py

Inference-only script for Smoke segmentation (IMAGE, FOLDER and VIDEO modes).
Goal: produce overlays that visualize per-pixel probabilities as precisely as possible
      (opacity per-pixel = prob * alpha_max). No binary masks are produced by default.

Defaults set to your requested paths/values; color selectable from CLI.
"""
import os
import argparse
import random
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from src.model import UNetSegmenter
from src.utils import load_checkpoint

# ------------------ model/load utilities ------------------
def load_model(path: str, device: torch.device):
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    model = UNetSegmenter(out_channels=1, pretrained=False).to(device)
    try:
        # try project utility first
        load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=device)
    except Exception:
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict):
            # common possible keys
            for k in ("model_state", "model_state_dict", "state_dict", "model"):
                if k in ckpt:
                    try:
                        model.load_state_dict(ckpt[k], strict=False)
                        break
                    except Exception:
                        pass
            else:
                try:
                    model.load_state_dict(ckpt, strict=False)
                except Exception:
                    pass
        else:
            try:
                model.load_state_dict(ckpt, strict=False)
            except Exception:
                pass
    model.eval()
    return model

# ------------------ preprocessing / helpers ------------------
def preprocess_image(path: str, size: int, device: torch.device):
    """
    Load image from disk, resize to (size,size) with bilinear, convert to tensor shape 1,C,H,W in [0,1]
    Returns (tensor, orig_rgb_numpy)
    """
    img = Image.open(path).convert("RGB")
    orig = np.array(img)  # H,W,3 RGB
    img_r = img.resize((size, size), resample=Image.BILINEAR)
    t = torch.from_numpy(np.array(img_r).transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)
    return t, orig

def preprocess_frame(frame_rgb: np.ndarray, size: int, device: torch.device):
    """
    Prepare a single frame (numpy RGB H,W,3 uint8) for model inference.
    Returns tensor 1,C,H,W on device.
    """
    img = Image.fromarray(frame_rgb)
    img_r = img.resize((size, size), resample=Image.BILINEAR)
    t = torch.from_numpy(np.array(img_r).transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)
    return t

def overlay_with_prob(orig_rgb: np.ndarray, prob_map: np.ndarray, color=(255,0,0), alpha_max: float = 1.0):
    """
    Create an overlay where each pixel is blended with `color` using alpha = prob * alpha_max.
    orig_rgb: H,W,3 uint8 (RGB)
    prob_map: H,W float in [0,1] (same H,W as orig) OR will be resized to orig (float interpolation)
    color: tuple (R,G,B)
    alpha_max: float in [0,1], maximum opacity for prob==1

    Returns RGB uint8 image.
    """
    # ensure input types
    orig = orig_rgb.copy().astype(np.float32)
    # resize prob_map to original image size using float32 interpolation (no quantization)
    if prob_map.dtype != np.float32:
        prob_map = prob_map.astype('float32')
    if prob_map.shape != orig.shape[:2]:
        prob_map = cv2.resize(prob_map, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

    # clamp and compute per-pixel alpha
    prob_map = np.clip(prob_map, 0.0, 1.0)
    alpha_map = prob_map * float(alpha_max)  # H,W
    alpha_map_3 = np.repeat(alpha_map[:, :, None], 3, axis=2)  # H,W,3
    color_arr = np.array(color, dtype=np.float32)[None, None, :]  # 1,1,3

    blended = (1.0 - alpha_map_3) * orig + alpha_map_3 * color_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def save_prob_png(prob_map: np.ndarray, out_path: str, cmap='magma'):
    """
    Save a probability heatmap PNG (visualization) using matplotlib colormap.
    prob_map expected in [0,1], dtype float32.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(prob_map, cmap=cm.get_cmap(cmap))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# ------------------ single-image mode ------------------
def mode_image(model, image_path, size, device, out_dir,
               save_probmaps=False, save_probpng=False, save_overlay=True, alpha_max=1.0, color=(255,0,0)):
    os.makedirs(out_dir, exist_ok=True)
    t, orig = preprocess_image(image_path, size, device)  # t: 1,C,H,W
    with torch.no_grad():
        try:
            with torch.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(t)
        except Exception:
            logits = model(t)
        probs = torch.sigmoid(logits)  # 1,1,Hp,Wp

    prob_np = probs.squeeze(0).squeeze(0).detach().cpu().numpy()  # Hp x Wp floats [0,1]

    stem = Path(image_path).stem
    probs_dir = Path(out_dir) / "probmaps"
    overlays_dir = Path(out_dir) / "overlays"
    if save_probmaps:
        probs_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(probs_dir / (stem + "_prob.npy")), prob_np.astype('float32'))
    if save_probpng:
        probs_dir.mkdir(parents=True, exist_ok=True)
        resized = cv2.resize(prob_np, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        save_prob_png(resized, str(probs_dir / (stem + "_prob.png")))

    if save_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)
        overlay_img = overlay_with_prob(orig, prob_np, color=color, alpha_max=alpha_max)
        cv2.imwrite(str(overlays_dir / (stem + "_overlay.png")), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    print(f"✅ Saved overlay for {image_path} -> {overlays_dir if save_overlay else out_dir}")

# ------------------ folder (batch) mode ------------------
def mode_folder(model, image_dir, out_dir, size, device, batch_size=8,
                sample_n=None, seed=42, save_probmaps=False, save_probpng=False,
                save_overlay=True, alpha_max=1.0, color=(255,0,0), preprocess_fn=None):
    model.to(device)
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    probs_dir = Path(out_dir) / "probmaps"
    overlays_dir = Path(out_dir) / "overlays"

    all_paths = sorted([p for p in Path(image_dir).glob("*") if p.suffix.lower() in ('.jpg', '.png', '.jpeg', '.tif', '.tiff')])
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
            orig_rgbs = []
            orig_sizes = []
            for p in batch_paths:
                pil = Image.open(p).convert('RGB')
                orig = np.array(pil)
                orig_rgbs.append(orig)
                orig_sizes.append(orig.shape[:2])  # H,W
                if preprocess_fn is not None:
                    t = preprocess_fn(str(p), size, device)
                else:
                    t, _ = preprocess_image(str(p), size, device)
                tensors.append(t)
            batch = torch.cat(tensors, dim=0)  # B,C,H,W

            try:
                with torch.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = model(batch)
            except Exception:
                logits = model(batch)
            probs = torch.sigmoid(logits)  # B,1,Hp,Wp
            probs_cpu = probs.detach().cpu().numpy()  # B,1,Hp,Wp

            for j, p in enumerate(batch_paths):
                base = p.stem
                prob = probs_cpu[j, 0]  # Hp x Wp float32
                h0, w0 = orig_sizes[j]

                if save_probmaps:
                    probs_dir.mkdir(exist_ok=True, parents=True)
                    np.save(str(probs_dir / (base + "_prob.npy")), prob.astype('float32'))

                if save_probpng:
                    probs_dir.mkdir(exist_ok=True, parents=True)
                    prob_vis = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    save_prob_png(prob_vis, str(probs_dir / (base + "_prob.png")))

                if save_overlay:
                    overlay_img = overlay_with_prob(orig_rgbs[j], prob, color=color, alpha_max=alpha_max)
                    overlays_dir.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(overlays_dir / (base + "_overlay.png")), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

                # brief logging
                meanp = float(np.mean(cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)))
                p90 = float(np.percentile(cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR), 90))
                print(f"[INF] {p.name} mean_prob={meanp:.4f} p90={p90:.4f}")

    print("Batch inference completed. Results in:", out_dir)

# ------------------ video mode ------------------
def mode_video(model, video_path, out_dir, size, device, save_overlay=True, alpha_max=1.0, color=(255,0,0), fps_limit=None):
    """
    Read video frames, predict probabilities per frame, create overlay per frame with per-pixel alpha = prob*alpha_max,
    and write resulting video preserving original resolution and fps.

    fps_limit: if set, will cap output fps to this value (useful if reading has inconsistent fps).
    """
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    overlays_dir = Path(out_dir) / "video_overlay"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps_limit is not None:
        fps = min(fps, float(fps_limit))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    basename = Path(video_path).stem
    out_file = str(Path(out_dir) / f"{basename}_prob_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
    pbar = tqdm(total=total, desc=f"Video {basename}", unit='frame')
    frame_idx = 0
    start_time = time.time()

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            # convert to RGB
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # preprocess frame to model input size
            t = preprocess_frame(rgb, size, device)  # 1,C,H,W
            try:
                with torch.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = model(t)
            except Exception:
                logits = model(t)
            probs = torch.sigmoid(logits)  # 1,1,Hp,Wp
            prob = probs.squeeze(0).squeeze(0).detach().cpu().numpy()  # Hp,Wp float32

            # create overlay on original resolution (float resize)
            overlay_img = overlay_with_prob(rgb, prob, color=color, alpha_max=alpha_max)
            # convert RGB->BGR for writer
            out_frame = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            writer.write(out_frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    print(f"\n✅ Video saved -> {out_file}")
    print(f"frames: {frame_idx}/{total} time: {total_time:.1f}s avg FPS: {avg_fps:.2f}")

# ------------------ CLI ------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Inference-only script for Smoke segmentation (image/folder/video)')
    p.add_argument('--mode', choices=['image', 'folder', 'video'], default="video")
    p.add_argument('--model_path', type=str, default=r"/home/nicola/Scaricati/checkpointssept22/smoke_best.pth")
    p.add_argument('--image_path', type=str, default=r"/home/nicola/Scrivania/test image from the net/20680295374_7af01a40b6_o.jpg")
    p.add_argument('--image_dir', type=str, default=r"/home/nicola/Scrivania/test image from the net")
    p.add_argument('--video_path', type=str, default=r"/home/nicola/Scrivania/test image from the net/wildfire_1.mp4")
    p.add_argument('--out_dir', type=str, default=r"/home/nicola/Scaricati/checkpointssept22/infer_prob")
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--threshold', type=float, default=0, help='(present for compatibility; not used for overlays)')
    p.add_argument('--no_overlay', action='store_true', default=False, help='Do not save overlay images')
    # color: accept names or R,G,B
    p.add_argument('--color', type=str, default='magenta', help="Overlay color: name (red/green/blue/yellow/cyan/magenta) or 'R,G,B' (e.g. 255,128,0)")
    # folder options
    p.add_argument('--batch_size', type=int, default=8, help='Batch size for folder inference')
    p.add_argument('--sample_n', type=int, default=None, help='If set, pseudo-randomly sample N images from folder')
    p.add_argument('--seed', type=int, default=42, help='Random seed for sampling images')
    p.add_argument('--save_probmaps', action='store_true', help='Also save raw .npy probability maps (float32)')
    p.add_argument('--save_probpng', action='store_true', help='Also save a colored PNG visualization of probmap (matplotlib colormap)')
    p.add_argument('--alpha_max', type=float, default=0.9, help='Maximum overlay opacity for prob==1.0 (0..1)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)

    # parse color: allow named or tuple
    color_map = {
        'red': (255,0,0),
        'green': (0,255,0),
        'blue': (0,0,255),
        'yellow': (255,255,0),
        'cyan': (0,255,255),
        'magenta': (255,0,255)
    }
    color = None
    if isinstance(args.color, str):
        s = args.color.strip().lower()
        if s in color_map:
            color = color_map[s]
        else:
            # try parse R,G,B
            try:
                parts = [int(x) for x in args.color.split(',')]
                if len(parts) == 3 and all(0 <= v <= 255 for v in parts):
                    color = tuple(parts)
            except Exception:
                color = None
    if color is None:
        color = (255,0,0)  # fallback red

    if args.mode == 'image':
        assert args.image_path, '--image_path required for image mode'
        mode_image(model, args.image_path, args.size, device, args.out_dir,
                   save_probmaps=args.save_probmaps, save_probpng=args.save_probpng,
                   save_overlay=(not args.no_overlay), alpha_max=args.alpha_max, color=color)
    elif args.mode == 'folder':
        assert args.image_dir, '--image_dir required for folder mode'
        mode_folder(model, args.image_dir, args.out_dir, args.size, device,
                    batch_size=args.batch_size, sample_n=args.sample_n, seed=args.seed,
                    save_probmaps=args.save_probmaps, save_probpng=args.save_probpng,
                    save_overlay=(not args.no_overlay), alpha_max=args.alpha_max, color=color)
    else:
        # video mode
        assert args.video_path, '--video_path required for video mode'
        mode_video(model, args.video_path, args.out_dir, args.size, device,
                   save_overlay=(not args.no_overlay), alpha_max=args.alpha_max, color=color)

