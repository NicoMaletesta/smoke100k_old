#!/usr/bin/env python3
"""
Testing script for Smoke segmentation (REQUIRES ground truth masks).
Now computes three measurement modes per image:
  1) Binary GT @127 (GT > 127) vs prediction binarized at --threshold
  2) Binary GT >1 (any non-zero) vs prediction binarized at --threshold
  3) Soft metrics: GT float in [0,1] vs prediction probabilities (no threshold)

Outputs:
 - per-image CSV: metrics_three_modes.csv (one row per image)
 - folder recap CSV: metrics_three_modes_summary.csv (one row per mode, with mean and std)
 - prints a human-readable table recap to stdout for both folder and image modes
 - image mode: saves two overlay PNGs in out_dir (overlay_gt127_<stem>.png, overlay_gt1_<stem>.png)
"""
import os
import csv
from pathlib import Path
import argparse
import time
import random
import math

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.model import UNetSegmenter
from src.utils import (
    load_checkpoint,
    metrics_at_thresholds,
    iou_score_proba_soft,
    dice_score_proba
)

# ------------------ model/load utilities ------------------
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

# ------------------ preprocessing / helpers ------------------
def preprocess_image(path: str, size: int, device: torch.device):
    img = Image.open(path).convert('RGB')
    orig = np.array(img)
    img_r = img.resize((size,size), resample=Image.BILINEAR)
    t = torch.from_numpy(np.array(img_r).transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)
    return t, orig

def load_mask_raw_and_soft(mask_path: str, size: int):
    """
    Load mask raw (grayscale) and produce:
      - arr_byte_resized: uint8 scaled to 0..255 resized to (size,size) (for binary thresholds >127 and >1)
      - arr_soft_norm: float32 normalized to [0,1] resized with bilinear (for soft metrics)
    """
    m = Image.open(mask_path).convert('L')
    arr = np.array(m).astype(np.float32)
    maxv = arr.max() if arr.size > 0 else 1.0

    # normalize to 0..255 for binary threshold operations
    if maxv <= 1.0:
        arr_byte = (arr * 255.0).astype(np.uint8)
    elif maxv <= 255.0:
        arr_byte = arr.astype(np.uint8)
    else:
        arr_byte = np.clip((arr / float(maxv)) * 255.0, 0, 255).astype(np.uint8)

    # For soft: resize using bilinear and normalize to [0,1]
    mr = m.resize((size,size), resample=Image.BILINEAR)
    arr_soft = np.array(mr).astype(np.float32)
    mx = arr_soft.max() if arr_soft.size>0 else 1.0
    if mx <= 1.0:
        arr_soft_norm = arr_soft
    elif mx <= 255.0:
        arr_soft_norm = arr_soft / 255.0
    else:
        arr_soft_norm = arr_soft / 65535.0
    arr_soft_norm = np.clip(arr_soft_norm, 0.0, 1.0)

    # arr_byte resized (nearest for binary masks)
    arr_byte_resized = cv2.resize(arr_byte, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    return arr_byte_resized, arr_soft_norm

def to_tensor_shape(arr, device):
    """Convert HxW numpy array (float or uint8) to torch tensor shape 1,1,H,W on specified device"""
    t = torch.from_numpy(arr.astype(np.float32))
    t = t.unsqueeze(0).unsqueeze(0).to(device)
    return t

def compute_soft_precision_recall_f1(probs_cpu, gt_soft):
    """
    Compute 'soft' precision/recall/f1:
      TP_soft = sum(p * gt)
      P_pred = sum(p)
      P_true = sum(gt)
    """
    p = probs_cpu.detach().cpu().numpy().astype(np.float64).ravel()
    g = gt_soft.detach().cpu().numpy().astype(np.float64).ravel()
    tp = float((p * g).sum())
    pred_pos = float(p.sum())
    true_pos = float(g.sum())

    if pred_pos == 0.0:
        prec = 1.0 if true_pos == 0.0 else 0.0
    else:
        prec = tp / pred_pos

    if true_pos == 0.0:
        rec = 1.0 if pred_pos == 0.0 else 0.0
    else:
        rec = tp / true_pos

    if (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = 0.0
    return prec, rec, f1

def safe_f1_from_prec_rec(prec, rec):
    if (prec + rec) > 0:
        return 2.0 * prec * rec / (prec + rec)
    return 0.0

def overlay_and_save(orig_np, pred_mask_rs, gt_mask_rs, out_path, alpha=0.45):
    """
    Create overlay image:
     - pred_mask_rs: HxW binary (0/1) at original image resolution
     - gt_mask_rs: HxW binary (0/1) at original image resolution
     - orig_np: HxWx3 uint8
    Pred is displayed as semi-transparent red; GT contours drawn in green.
    """
    overlay = orig_np.copy().astype(np.float32)
    # color for prediction: red (BGR: (0,0,255) in cv2)
    red = np.array([255, 0, 0], dtype=np.float32)  # we will save as RGB at the end (orig is RGB)
    mask_pred = (pred_mask_rs > 0).astype(bool)
    # blend red where predicted
    overlay[mask_pred] = overlay[mask_pred] * (1.0 - alpha) + red * alpha
    overlay = overlay.astype(np.uint8)

    # draw GT contours in green (RGB)
    gt_uint8 = (gt_mask_rs > 0).astype(np.uint8) * 255
    # cv2 expects BGR, but our image is RGB; convert temporarily
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay_bgr, contours, -1, (0,255,0), thickness=2)  # green in BGR
    # convert back to RGB
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))  # cv2.imwrite expects BGR, so convert
    # (we saved via cv2 so we convert back to BGR for write)

# ------------------ folder / evaluation ------------------
def mode_folder(model, image_dir, mask_dir, size, device, out_dir,
                thr=0.5, sample_n=None, seed=42, batch=False, batch_size=8, save_probmaps=False):
    image_paths = sorted([p for p in Path(image_dir).glob('*') if p.suffix.lower() in ('.jpg','.png','.jpeg')])
    mask_map = {p.stem: p for p in Path(mask_dir).glob('*')} if Path(mask_dir).exists() else {}
    valid_pairs = [(p, mask_map[p.stem]) for p in image_paths if p.stem in mask_map]
    if len(valid_pairs) == 0:
        print("[WARN] No image/mask pairs found. Check mask_dir and image_dir.")
        return

    if sample_n is not None and sample_n < len(valid_pairs):
        random.seed(seed)
        valid_pairs = random.sample(valid_pairs, sample_n)
    valid_pairs = sorted(valid_pairs, key=lambda x: x[0].name)

    per_image_csv = Path(out_dir) / 'metrics_three_modes.csv'
    summary_csv = Path(out_dir) / 'metrics_three_modes_summary.csv'
    per_image_fields = [
        'file',
        # MODE A: binary GT @127 (binary preds @thr)
        'iou_127','dice_127','precision_127','recall_127','f1_127',
        # MODE B: binary GT >1 (binary preds @thr)
        'iou_nonzero','dice_nonzero','precision_nonzero','recall_nonzero','f1_nonzero',
        # MODE C: soft GT vs prob preds (no threshold)
        'iou_soft','dice_soft','precision_soft','recall_soft','f1_soft'
    ]

    # lists for stats
    lists = {k: [] for k in per_image_fields if k != 'file'}
    n = 0

    probs_dir = Path(out_dir) / "probmaps"
    if save_probmaps:
        probs_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(per_image_csv,'w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_image_fields)
        writer.writeheader()

        if not batch:
            with torch.no_grad():
                for img_p, mask_p in tqdm(valid_pairs, desc='Folder testing'):
                    inp, orig = preprocess_image(str(img_p), size, device)
                    logits = model(inp)
                    probs = torch.sigmoid(logits)  # 1,1,H,W
                    probs_cpu = probs.detach().cpu()  # shape 1,1,H,W

                    if save_probmaps:
                        prob = probs_cpu[0,0].numpy()
                        h0, w0 = orig.shape[:2]
                        prob_resized = cv2.resize((prob * 255.0).astype('uint8'), (w0, h0),
                                                  interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
                        np.save(str(probs_dir / (img_p.stem + "_prob.npy")), prob_resized)

                    # load mask raw & soft at target size
                    arr_byte_resized, arr_soft = load_mask_raw_and_soft(str(mask_p), size)

                    # ---------- MODE A: GT binarized @127 ----------
                    gt_127 = (arr_byte_resized > 127).astype(np.uint8)  # HxW
                    gt_127_t = to_tensor_shape(gt_127, device=torch.device('cpu'))  # 1,1,H,W on cpu for metrics
                    metrics_127 = metrics_at_thresholds(probs_cpu, gt_127_t, thresholds=[thr])
                    mm127 = metrics_127[thr]
                    iou_127 = mm127['iou']; dice_127 = mm127['dice']; prec_127 = mm127['precision']; rec_127 = mm127['recall']
                    f1_127 = safe_f1_from_prec_rec(prec_127, rec_127)

                    # ---------- MODE B: GT binarized >1 (non-zero) ----------
                    gt_nz = (arr_byte_resized > 1).astype(np.uint8)
                    gt_nz_t = to_tensor_shape(gt_nz, device=torch.device('cpu'))
                    metrics_nz = metrics_at_thresholds(probs_cpu, gt_nz_t, thresholds=[thr])
                    mmnz = metrics_nz[thr]
                    iou_nz = mmnz['iou']; dice_nz = mmnz['dice']; prec_nz = mmnz['precision']; rec_nz = mmnz['recall']
                    f1_nz = safe_f1_from_prec_rec(prec_nz, rec_nz)

                    # ---------- MODE C: Soft ----------
                    gt_soft_t = to_tensor_shape(arr_soft, device=torch.device('cpu'))  # 1,1,H,W float in [0,1]
                    iou_s = iou_score_proba_soft(probs_cpu, gt_soft_t)
                    dice_s = dice_score_proba(probs_cpu, gt_soft_t)
                    prec_s, rec_s, f1_s = compute_soft_precision_recall_f1(probs_cpu, gt_soft_t)

                    writer.writerow({
                        'file': img_p.name,
                        'iou_127': iou_127, 'dice_127': dice_127, 'precision_127': prec_127, 'recall_127': rec_127, 'f1_127': f1_127,
                        'iou_nonzero': iou_nz, 'dice_nonzero': dice_nz, 'precision_nonzero': prec_nz, 'recall_nonzero': rec_nz, 'f1_nonzero': f1_nz,
                        'iou_soft': iou_s, 'dice_soft': dice_s, 'precision_soft': prec_s, 'recall_soft': rec_s, 'f1_soft': f1_s
                    })

                    # append to lists
                    lists['iou_127'].append(iou_127); lists['dice_127'].append(dice_127); lists['precision_127'].append(prec_127); lists['recall_127'].append(rec_127); lists['f1_127'].append(f1_127)
                    lists['iou_nonzero'].append(iou_nz); lists['dice_nonzero'].append(dice_nz); lists['precision_nonzero'].append(prec_nz); lists['recall_nonzero'].append(rec_nz); lists['f1_nonzero'].append(f1_nz)
                    lists['iou_soft'].append(iou_s); lists['dice_soft'].append(dice_s); lists['precision_soft'].append(prec_s); lists['recall_soft'].append(rec_s); lists['f1_soft'].append(f1_s)

                    n += 1

        else:
            # batch processing
            model.to(device)
            model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, len(valid_pairs), batch_size), desc='Batch testing'):
                    batch_pairs = valid_pairs[i:i+batch_size]
                    tensors = []
                    orig_sizes = []
                    masks_info = []
                    for img_p, mask_p in batch_pairs:
                        t, orig = preprocess_image(str(img_p), size, device)
                        tensors.append(t)
                        orig_sizes.append(orig.shape[:2])
                        arr_byte_resized, arr_soft = load_mask_raw_and_soft(str(mask_p), size)
                        masks_info.append((arr_byte_resized, arr_soft))
                    batch_tensor = torch.cat(tensors, dim=0)  # B,C,H,W

                    try:
                        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                            logits = model(batch_tensor)
                    except Exception:
                        logits = model(batch_tensor)
                    probs = torch.sigmoid(logits)  # B,1,H,W
                    probs_cpu = probs.detach().cpu()

                    for j, (img_p, mask_p) in enumerate(batch_pairs):
                        prob_j = probs_cpu[j].unsqueeze(0)  # 1,1,H,W
                        arr_byte_resized, arr_soft = masks_info[j]

                        # MODE A
                        gt_127 = (arr_byte_resized > 127).astype(np.uint8)
                        gt_127_t = to_tensor_shape(gt_127, device=torch.device('cpu'))
                        metrics_127 = metrics_at_thresholds(prob_j, gt_127_t, thresholds=[thr])
                        mm127 = metrics_127[thr]
                        iou_127 = mm127['iou']; dice_127 = mm127['dice']; prec_127 = mm127['precision']; rec_127 = mm127['recall']
                        f1_127 = safe_f1_from_prec_rec(prec_127, rec_127)

                        # MODE B
                        gt_nz = (arr_byte_resized > 1).astype(np.uint8)
                        gt_nz_t = to_tensor_shape(gt_nz, device=torch.device('cpu'))
                        metrics_nz = metrics_at_thresholds(prob_j, gt_nz_t, thresholds=[thr])
                        mmnz = metrics_nz[thr]
                        iou_nz = mmnz['iou']; dice_nz = mmnz['dice']; prec_nz = mmnz['precision']; rec_nz = mmnz['recall']
                        f1_nz = safe_f1_from_prec_rec(prec_nz, rec_nz)

                        # MODE C (soft)
                        gt_soft_t = to_tensor_shape(arr_soft, device=torch.device('cpu'))
                        iou_s = iou_score_proba_soft(prob_j, gt_soft_t)
                        dice_s = dice_score_proba(prob_j, gt_soft_t)
                        prec_s, rec_s, f1_s = compute_soft_precision_recall_f1(prob_j, gt_soft_t)

                        writer.writerow({
                            'file': img_p.name,
                            'iou_127': iou_127, 'dice_127': dice_127, 'precision_127': prec_127, 'recall_127': rec_127, 'f1_127': f1_127,
                            'iou_nonzero': iou_nz, 'dice_nonzero': dice_nz, 'precision_nonzero': prec_nz, 'recall_nonzero': rec_nz, 'f1_nonzero': f1_nz,
                            'iou_soft': iou_s, 'dice_soft': dice_s, 'precision_soft': prec_s, 'recall_soft': rec_s, 'f1_soft': f1_s
                        })

                        # append
                        lists['iou_127'].append(iou_127); lists['dice_127'].append(dice_127); lists['precision_127'].append(prec_127); lists['recall_127'].append(rec_127); lists['f1_127'].append(f1_127)
                        lists['iou_nonzero'].append(iou_nz); lists['dice_nonzero'].append(dice_nz); lists['precision_nonzero'].append(prec_nz); lists['recall_nonzero'].append(rec_nz); lists['f1_nonzero'].append(f1_nz)
                        lists['iou_soft'].append(iou_s); lists['dice_soft'].append(dice_s); lists['precision_soft'].append(prec_s); lists['recall_soft'].append(rec_s); lists['f1_soft'].append(f1_s)

                        n += 1

    # build summary (means and stds)
    summary_rows = []
    # helper to safe mean/std
    def ms(key):
        arr = np.array(lists[key], dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0
        return float(np.mean(arr)), float(np.std(arr, ddof=0))

    # Mode A
    iou_m, iou_s = ms('iou_127')
    dice_m, dice_s = ms('dice_127')
    prec_m, prec_s = ms('precision_127')
    rec_m, rec_s = ms('recall_127')
    f1_m, f1_s = ms('f1_127')
    summary_rows.append({
        'mode': 'GT@127 (binary)', 'count': n,
        'iou_mean': iou_m, 'iou_std': iou_s,
        'dice_mean': dice_m, 'dice_std': dice_s,
        'precision_mean': prec_m, 'precision_std': prec_s,
        'recall_mean': rec_m, 'recall_std': rec_s,
        'f1_mean': f1_m, 'f1_std': f1_s
    })

    # Mode B
    iou_m, iou_s = ms('iou_nonzero')
    dice_m, dice_s = ms('dice_nonzero')
    prec_m, prec_s = ms('precision_nonzero')
    rec_m, rec_s = ms('recall_nonzero')
    f1_m, f1_s = ms('f1_nonzero')
    summary_rows.append({
        'mode': 'GT>1 (non-zero binary)', 'count': n,
        'iou_mean': iou_m, 'iou_std': iou_s,
        'dice_mean': dice_m, 'dice_std': dice_s,
        'precision_mean': prec_m, 'precision_std': prec_s,
        'recall_mean': rec_m, 'recall_std': rec_s,
        'f1_mean': f1_m, 'f1_std': f1_s
    })

    # Mode C
    iou_m, iou_s = ms('iou_soft')
    dice_m, dice_s = ms('dice_soft')
    prec_m, prec_s = ms('precision_soft')
    rec_m, rec_s = ms('recall_soft')
    f1_m, f1_s = ms('f1_soft')
    summary_rows.append({
        'mode': 'Soft (continuous)', 'count': n,
        'iou_mean': iou_m, 'iou_std': iou_s,
        'dice_mean': dice_m, 'dice_std': dice_s,
        'precision_mean': prec_m, 'precision_std': prec_s,
        'recall_mean': rec_m, 'recall_std': rec_s,
        'f1_mean': f1_m, 'f1_std': f1_s
    })

    # write summary CSV (with mean and std columns)
    summary_fields = ['mode','count',
                      'iou_mean','iou_std','dice_mean','dice_std',
                      'precision_mean','precision_std','recall_mean','recall_std',
                      'f1_mean','f1_std']
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # print pretty table to stdout (mean ± std)
    print("\n=== SUMMARY TABLE ===")
    colfmt = "{:<28} {:>6} {:>18} {:>18} {:>18} {:>18} {:>12}"
    print(colfmt.format("mode","n","IoU (mean±std)","Dice (mean±std)","Precision (mean±std)","Recall (mean±std)","F1 (mean±std)"))
    print("-"*120)
    for r in summary_rows:
        def ms_str(mean, std):
            return f"{mean:.4f}±{std:.4f}"
        print(colfmt.format(
            r['mode'][:28],
            str(r['count']),
            ms_str(r['iou_mean'], r['iou_std']),
            ms_str(r['dice_mean'], r['dice_std']),
            ms_str(r['precision_mean'], r['precision_std']),
            ms_str(r['recall_mean'], r['recall_std']),
            ms_str(r['f1_mean'], r['f1_std'])
        ))
    print(f"\nPer-image metrics saved -> {per_image_csv}")
    print(f"Summary saved -> {summary_csv}\n")
    return summary_rows

# ------------------ single image mode (prints same recap for that image) ------------------
def mode_image(model, image_path, mask_path, size, device, thr=0.5, save_probmap=False, out_dir="./"):
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        raise FileNotFoundError("image or mask not found")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    inp, orig = preprocess_image(image_path, size, device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)
        probs_cpu = probs.detach().cpu()

    arr_byte_resized, arr_soft = load_mask_raw_and_soft(mask_path, size)

    # MODE A
    gt_127 = (arr_byte_resized > 127).astype(np.uint8)
    gt_127_t = to_tensor_shape(gt_127, device=torch.device('cpu'))
    metrics_127 = metrics_at_thresholds(probs_cpu, gt_127_t, thresholds=[thr])
    mm127 = metrics_127[thr]
    iou_127 = mm127['iou']; dice_127 = mm127['dice']; prec_127 = mm127['precision']; rec_127 = mm127['recall']
    f1_127 = safe_f1_from_prec_rec(prec_127, rec_127)

    # MODE B
    gt_nz = (arr_byte_resized > 1).astype(np.uint8)
    gt_nz_t = to_tensor_shape(gt_nz, device=torch.device('cpu'))
    metrics_nz = metrics_at_thresholds(probs_cpu, gt_nz_t, thresholds=[thr])
    mmnz = metrics_nz[thr]
    iou_nz = mmnz['iou']; dice_nz = mmnz['dice']; prec_nz = mmnz['precision']; rec_nz = mmnz['recall']
    f1_nz = safe_f1_from_prec_rec(prec_nz, rec_nz)

    # MODE C soft
    gt_soft_t = to_tensor_shape(arr_soft, device=torch.device('cpu'))
    iou_s = iou_score_proba_soft(probs_cpu, gt_soft_t)
    dice_s = dice_score_proba(probs_cpu, gt_soft_t)
    prec_s, rec_s, f1_s = compute_soft_precision_recall_f1(probs_cpu, gt_soft_t)

    # print table
    print("\n=== IMAGE METRICS SUMMARY ===")
    colfmt = "{:<28} {:>8} {:>8} {:>10} {:>8} {:>8}"
    print(colfmt.format("mode","IoU","Dice","Precision","Recall","F1"))
    print("-"*70)
    print(colfmt.format("GT@127 (binary)", f"{iou_127:.4f}", f"{dice_127:.4f}", f"{prec_127:.4f}", f"{rec_127:.4f}", f"{f1_127:.4f}"))
    print(colfmt.format("GT>1 (non-zero)", f"{iou_nz:.4f}", f"{dice_nz:.4f}", f"{prec_nz:.4f}", f"{rec_nz:.4f}", f"{f1_nz:.4f}"))
    print(colfmt.format("Soft (continuous)", f"{iou_s:.4f}", f"{dice_s:.4f}", f"{prec_s:.4f}", f"{rec_s:.4f}", f"{f1_s:.4f}"))
    print()

    # optionally save probmap (resized to original)
    orig_h, orig_w = orig.shape[:2]
    if save_probmap:
        prob = probs_cpu[0,0].numpy()
        prob_resized = cv2.resize((prob * 255.0).astype('uint8'), (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
        np.save(str(Path(out_dir)/(Path(image_path).stem + "_prob.npy")), prob_resized)

    # create and save two overlay images (GT@127 and GT>1)
    # predicted binary mask at network resolution:
    pred_bin_size = (probs_cpu[0,0].numpy() > thr).astype(np.uint8)  # HxW (network size)
    # resize pred and gt masks to original image resolution for overlay
    pred_rs = cv2.resize(pred_bin_size.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    gt127_rs = cv2.resize((gt_127.astype(np.uint8)), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    gtnz_rs = cv2.resize((gt_nz.astype(np.uint8)), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # save overlays
    stem = Path(image_path).stem
    out1 = Path(out_dir) / f"overlay_{stem}_gt127.png"
    out2 = Path(out_dir) / f"overlay_{stem}_gt1.png"

    # orig is RGB, overlay_and_save expects RGB
    overlay_and_save(orig, pred_rs, gt127_rs, out1)
    overlay_and_save(orig, pred_rs, gtnz_rs, out2)

    print(f"Overlay images saved -> {out1}, {out2}")

    # return structured summary
    return [
        {'mode': 'GT@127 (binary)', 'iou': iou_127, 'dice': dice_127, 'precision': prec_127, 'recall': rec_127, 'f1': f1_127},
        {'mode': 'GT>1 (non-zero)', 'iou': iou_nz, 'dice': dice_nz, 'precision': prec_nz, 'recall': rec_nz, 'f1': f1_nz},
        {'mode': 'Soft (continuous)', 'iou': iou_s, 'dice': dice_s, 'precision': prec_s, 'recall': rec_s, 'f1': f1_s}
    ]

# ------------------ CLI ------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Testing script for Smoke segmentation (three measurement modes).')
    p.add_argument('--mode', choices=['image','folder'], default="folder")
    p.add_argument('--model_path', type=str, default=r"/home/nicola/Documenti/testsmoke100k/smoke_best.pth")
    p.add_argument('--image_path', type=str, default=r"/home/nicola/Scaricati/smoke100kprocessed/test/images/smoke100k-H_test_000774.png")
    p.add_argument('--mask_path', type=str, default= r"/home/nicola/Scaricati/smoke100kprocessed/test/masks/smoke100k-H_test_000774.png")
    p.add_argument('--image_dir', type=str, default=r"/home/nicola/Scaricati/smoke100kprocessed/test/images")
    p.add_argument('--mask_dir', type=str, default=r"/home/nicola/Scaricati/smoke100kprocessed/test/masks")
    p.add_argument('--out_dir', type=str, default=r"./test_out")
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--batch', action='store_true', help='Use batch mode for folder testing (faster)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--sample_n', type=int, default=2000, help='If set, pseudo-randomly sample N image/mask pairs')
    p.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    p.add_argument('--save_probmaps', action='store_true', help='Save probability maps (.npy) for each image')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)

    if args.mode == 'folder':
        mode_folder(model, args.image_dir, args.mask_dir, args.size, device, args.out_dir,
                    thr=args.threshold, sample_n=args.sample_n, seed=args.seed,
                    batch=args.batch, batch_size=args.batch_size, save_probmaps=args.save_probmaps)
    else:
        # image mode: requires --image_path and --mask_path
        if not args.image_path or not args.mask_path:
            raise ValueError("For image mode provide --image_path and --mask_path")
        mode_image(model, args.image_path, args.mask_path, args.size, device, thr=args.threshold, save_probmap=args.save_probmaps, out_dir=args.out_dir)
