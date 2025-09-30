#!/usr/bin/env python3
"""
infer_prob_all_video.py

Batch video inference for smoke segmentation.

Input:
 - --model_path    : path al checkpoint del modello (carica con load_checkpoint se disponibile)
 - --videos_dir    : cartella contenente video (estensioni comuni: mp4, avi, mov, mkv)
 - --out_dir       : cartella dove salvare i video output (uno per input, nome: <basename>_prob_overlay.mp4)

Comportamento:
 - usa lo stesso comportamento di inferenza/video overlay del tuo infer_probability.py
 - mantiene risoluzione e fps originali (opzionalmente limita fps con --fps_limit)
"""
import os
import time
import argparse
from pathlib import Path

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
    orig = orig_rgb.copy().astype(np.float32)
    if prob_map.dtype != np.float32:
        prob_map = prob_map.astype('float32')
    if prob_map.shape != orig.shape[:2]:
        prob_map = cv2.resize(prob_map, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

    prob_map = np.clip(prob_map, 0.0, 1.0)
    alpha_map = prob_map * float(alpha_max)  # H,W
    alpha_map_3 = np.repeat(alpha_map[:, :, None], 3, axis=2)  # H,W,3
    color_arr = np.array(color, dtype=np.float32)[None, None, :]  # 1,1,3

    blended = (1.0 - alpha_map_3) * orig + alpha_map_3 * color_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

# ------------------ single video processing ------------------
def process_video(model, video_path: str, out_dir: str, size: int, device: torch.device,
                  save_overlay: bool = True, alpha_max: float = 0.9, color=(255,0,0), fps_limit: float = None):
    """
    Process one video: run per-frame inference and write an overlay video to out_dir.
    Output filename: <basename>_prob_overlay.mp4
    """
    video_path = str(video_path)
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video {video_path}, skipping.")
        return None

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
    if not writer.isOpened():
        print(f"[WARN] Could not open writer for {out_file}. Skipping.")
        cap.release()
        return None

    pbar = tqdm(total=total, desc=f"Video {basename}", unit='frame')
    frame_idx = 0
    start_time = time.time()

    model.to(device)
    model.eval()
    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t = preprocess_frame(rgb, size, device)  # 1,C,H,W
            try:
                with torch.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = model(t)
            except Exception:
                logits = model(t)
            probs = torch.sigmoid(logits)  # 1,1,Hp,Wp
            prob = probs.squeeze(0).squeeze(0).detach().cpu().numpy()  # Hp,Wp float32

            overlay_img = overlay_with_prob(rgb, prob, color=color, alpha_max=alpha_max)
            out_frame = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            writer.write(out_frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    print(f"✅ Saved -> {out_file}  frames: {frame_idx}/{total} time: {total_time:.1f}s avg FPS: {avg_fps:.2f}")
    return out_file

# ------------------ batch over folder ------------------
def infer_all_videos(model, videos_dir: str, out_dir: str, size: int, device: torch.device,
                     alpha_max: float = 0.9, color=(255,0,0), fps_limit: float = None):
    p = Path(videos_dir)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"videos_dir non valido: {videos_dir}")

    exts = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    videos = sorted([x for x in p.iterdir() if x.is_file() and x.suffix in exts])
    if len(videos) == 0:
        print(f"[WARN] Nessun video trovato in {videos_dir} (estensioni cercate: {sorted(exts)})")
        return

    results = []
    for v in videos:
        print(f"\n--- Processing: {v.name} ---")
        try:
            out_path = process_video(model, str(v), out_dir, size, device,
                                     save_overlay=True, alpha_max=alpha_max, color=color, fps_limit=fps_limit)
            results.append((str(v), out_path))
        except Exception as e:
            print(f"[ERROR] Fallita elaborazione {v.name}: {e}")
    print("\nBatch completo. Risultati:")
    for inp, out in results:
        print(f" - {Path(inp).name} -> {out}")
    return results

# ------------------ CLI ------------------
def parse_color(s: str):
    color_map = {
        'red': (255,0,0),
        'green': (0,255,0),
        'blue': (0,0,255),
        'yellow': (255,255,0),
        'cyan': (0,255,255),
        'magenta': (255,0,255)
    }
    if s is None:
        return (255,0,0)
    s = s.strip().lower()
    if s in color_map:
        return color_map[s]
    try:
        parts = [int(x) for x in s.split(',')]
        if len(parts) == 3 and all(0 <= v <= 255 for v in parts):
            return tuple(parts)
    except Exception:
        pass
    return (255,0,0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch video inference (overlay probability) for all videos in a folder")
    parser.add_argument('--model_path', type=str,  default=r"/home/nicola/Documenti/smokedetector_2909/smoke_best.pth", help='Path to model checkpoint')
    parser.add_argument('--videos_dir', type=str, default=r"/home/nicola/Immagini/cut out video reg cal", help='Folder with videos to process')
    parser.add_argument('--out_dir', type=str, default= r"/home/nicola/Documenti/smokedetector_2909/infer_probmulti", help='Output folder to store result videos')
    parser.add_argument('--size', type=int, default=512, help='Model input size (square)')
    parser.add_argument('--alpha_max', type=float, default=0.9, help='Max overlay opacity for prob==1.0 (0..1)')
    parser.add_argument('--color', type=str, default='magenta', help="Overlay color name or 'R,G,B'")
    parser.add_argument('--fps_limit', type=float, default=None, help='Optionally cap output FPS to this value')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)
    color = parse_color(args.color)

    infer_all_videos(model, args.videos_dir, args.out_dir, args.size, device,
                     alpha_max=args.alpha_max, color=color, fps_limit=args.fps_limit)
