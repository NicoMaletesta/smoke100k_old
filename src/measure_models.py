#!/usr/bin/env python3
"""
measure_models.py

Run the "folder" evaluation (three measurement modes) for every .pth checkpoint
found in a directory. For each checkpoint:
  - runs folder-mode evaluation (same behavior as your testing script)
  - saves per-image CSV and per-model summary CSV in out_dir/<ckpt_stem>/
  - optionally save probmaps/overlays as configured

After all models are processed, prints and saves a global CSV recap with one row per (model, mode).

Defaults in argparse match those you provided in the testing script.
"""
import os
import csv
import time
import random
from pathlib import Path
import argparse
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from src.model import UNetSegmenter
from src.utils import load_checkpoint, metrics_at_thresholds, iou_score_proba_soft, dice_score_proba

# ------------------ helpers (copied/adapted from your testing script) ------------------
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

def preprocess_image(path: str, size: int, device: torch.device):
    img = Image.open(path).convert('RGB')
    orig = np.array(img)
    img_r = img.resize((size, size), resample=Image.BILINEAR)
    t = torch.from_numpy(np.array(img_r).transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)
    return t, orig

def load_mask_raw_and_soft(mask_path: str, size: int):
    m = Image.open(mask_path).convert('L')
    arr = np.array(m).astype(np.float32)
    maxv = arr.max() if arr.size > 0 else 1.0

    if maxv <= 1.0:
        arr_byte = (arr * 255.0).astype(np.uint8)
    elif maxv <= 255.0:
        arr_byte = arr.astype(np.uint8)
    else:
        arr_byte = np.clip((arr / float(maxv)) * 255.0, 0, 255).astype(np.uint8)

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

    arr_byte_resized = cv2.resize(arr_byte, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return arr_byte_resized, arr_soft_norm

def to_tensor_shape(arr, device):
    t = torch.from_numpy(arr.astype(np.float32))
    t = t.unsqueeze(0).unsqueeze(0).to(device)
    return t

def compute_soft_precision_recall_f1(probs_cpu, gt_soft):
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
    overlay = orig_np.copy().astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    mask_pred = (pred_mask_rs > 0).astype(bool)
    overlay[mask_pred] = overlay[mask_pred] * (1.0 - alpha) + red * alpha
    overlay = overlay.astype(np.uint8)

    gt_uint8 = (gt_mask_rs > 0).astype(np.uint8) * 255
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay_bgr, contours, -1, (0,255,0), thickness=2)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

# ------------------ adapted mode_folder (returns summary_rows like original) ------------------
def mode_folder_and_return_summary(model,
                image_dir, mask_dir, size, device, out_dir,
                thr=0.5, sample_n=None, seed=42, batch=False, batch_size=8, save_probmaps=False):
    """
    Wrapper of your mode_folder logic that returns the summary_rows.
    Saves per-image CSV and summary CSV in out_dir.
    """
    image_paths = sorted([p for p in Path(image_dir).glob('*') if p.suffix.lower() in ('.jpg','.png','.jpeg')])
    mask_map = {p.stem: p for p in Path(mask_dir).glob('*')} if Path(mask_dir).exists() else {}
    valid_pairs = [(p, mask_map[p.stem]) for p in image_paths if p.stem in mask_map]
    if len(valid_pairs) == 0:
        print("[WARN] No image/mask pairs found. Check mask_dir and image_dir.")
        return []

    if sample_n is not None and sample_n < len(valid_pairs):
        random.seed(seed)
        valid_pairs = random.sample(valid_pairs, sample_n)
    valid_pairs = sorted(valid_pairs, key=lambda x: x[0].name)

    per_image_csv = Path(out_dir) / 'metrics_three_modes.csv'
    summary_csv = Path(out_dir) / 'metrics_three_modes_summary.csv'
    per_image_fields = [
        'file',
        'iou_127','dice_127','precision_127','recall_127','f1_127',
        'iou_nonzero','dice_nonzero','precision_nonzero','recall_nonzero','f1_nonzero',
        'iou_soft','dice_soft','precision_soft','recall_soft','f1_soft'
    ]

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

                    arr_byte_resized, arr_soft = load_mask_raw_and_soft(str(mask_p), size)

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

                    writer.writerow({
                        'file': img_p.name,
                        'iou_127': iou_127, 'dice_127': dice_127, 'precision_127': prec_127, 'recall_127': rec_127, 'f1_127': f1_127,
                        'iou_nonzero': iou_nz, 'dice_nonzero': dice_nz, 'precision_nonzero': prec_nz, 'recall_nonzero': rec_nz, 'f1_nonzero': f1_nz,
                        'iou_soft': iou_s, 'dice_soft': dice_s, 'precision_soft': prec_s, 'recall_soft': rec_s, 'f1_soft': f1_s
                    })

                    lists['iou_127'].append(iou_127); lists['dice_127'].append(dice_127); lists['precision_127'].append(prec_127); lists['recall_127'].append(rec_127); lists['f1_127'].append(f1_127)
                    lists['iou_nonzero'].append(iou_nz); lists['dice_nonzero'].append(dice_nz); lists['precision_nonzero'].append(prec_nz); lists['recall_nonzero'].append(rec_nz); lists['f1_nonzero'].append(f1_nz)
                    lists['iou_soft'].append(iou_s); lists['dice_soft'].append(dice_s); lists['precision_soft'].append(prec_s); lists['recall_soft'].append(rec_s); lists['f1_soft'].append(f1_s)

                    n += 1

        else:
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
                    batch_tensor = torch.cat(tensors, dim=0)

                    try:
                        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                            logits = model(batch_tensor)
                    except Exception:
                        logits = model(batch_tensor)
                    probs = torch.sigmoid(logits)
                    probs_cpu = probs.detach().cpu()

                    for j, (img_p, mask_p) in enumerate(batch_pairs):
                        prob_j = probs_cpu[j].unsqueeze(0)
                        arr_byte_resized, arr_soft = masks_info[j]

                        gt_127 = (arr_byte_resized > 127).astype(np.uint8)
                        gt_127_t = to_tensor_shape(gt_127, device=torch.device('cpu'))
                        metrics_127 = metrics_at_thresholds(prob_j, gt_127_t, thresholds=[thr])
                        mm127 = metrics_127[thr]
                        iou_127 = mm127['iou']; dice_127 = mm127['dice']; prec_127 = mm127['precision']; rec_127 = mm127['recall']
                        f1_127 = safe_f1_from_prec_rec(prec_127, rec_127)

                        gt_nz = (arr_byte_resized > 1).astype(np.uint8)
                        gt_nz_t = to_tensor_shape(gt_nz, device=torch.device('cpu'))
                        metrics_nz = metrics_at_thresholds(prob_j, gt_nz_t, thresholds=[thr])
                        mmnz = metrics_nz[thr]
                        iou_nz = mmnz['iou']; dice_nz = mmnz['dice']; prec_nz = mmnz['precision']; rec_nz = mmnz['recall']
                        f1_nz = safe_f1_from_prec_rec(prec_nz, rec_nz)

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

                        lists['iou_127'].append(iou_127); lists['dice_127'].append(dice_127); lists['precision_127'].append(prec_127); lists['recall_127'].append(rec_127); lists['f1_127'].append(f1_127)
                        lists['iou_nonzero'].append(iou_nz); lists['dice_nonzero'].append(dice_nz); lists['precision_nonzero'].append(prec_nz); lists['recall_nonzero'].append(rec_nz); lists['f1_nonzero'].append(f1_nz)
                        lists['iou_soft'].append(iou_s); lists['dice_soft'].append(dice_s); lists['precision_soft'].append(prec_s); lists['recall_soft'].append(rec_s); lists['f1_soft'].append(f1_s)

                        n += 1

    # build summary rows (same format as your original mode_folder)
    summary_rows = []
    def ms(key):
        arr = np.array(lists[key], dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0
        return float(np.mean(arr)), float(np.std(arr, ddof=0))

    # Mode A
    iou_m, iou_s = ms('iou_127'); dice_m, dice_s = ms('dice_127')
    prec_m, prec_s = ms('precision_127'); rec_m, rec_s = ms('recall_127'); f1_m, f1_s = ms('f1_127')
    summary_rows.append({
        'mode': 'GT@127 (binary)', 'count': n,
        'iou_mean': iou_m, 'iou_std': iou_s,
        'dice_mean': dice_m, 'dice_std': dice_s,
        'precision_mean': prec_m, 'precision_std': prec_s,
        'recall_mean': rec_m, 'recall_std': rec_s,
        'f1_mean': f1_m, 'f1_std': f1_s
    })

    # Mode B
    iou_m, iou_s = ms('iou_nonzero'); dice_m, dice_s = ms('dice_nonzero')
    prec_m, prec_s = ms('precision_nonzero'); rec_m, rec_s = ms('recall_nonzero'); f1_m, f1_s = ms('f1_nonzero')
    summary_rows.append({
        'mode': 'GT>1 (non-zero binary)', 'count': n,
        'iou_mean': iou_m, 'iou_std': iou_s,
        'dice_mean': dice_m, 'dice_std': dice_s,
        'precision_mean': prec_m, 'precision_std': prec_s,
        'recall_mean': rec_m, 'recall_std': rec_s,
        'f1_mean': f1_m, 'f1_std': f1_s
    })

    # Mode C
    iou_m, iou_s = ms('iou_soft'); dice_m, dice_s = ms('dice_soft')
    prec_m, prec_s = ms('precision_soft'); rec_m, rec_s = ms('recall_soft'); f1_m, f1_s = ms('f1_soft')
    summary_rows.append({
        'mode': 'Soft (continuous)', 'count': n,
        'iou_mean': iou_m, 'iou_std': iou_s,
        'dice_mean': dice_m, 'dice_std': dice_s,
        'precision_mean': prec_m, 'precision_std': prec_s,
        'recall_mean': rec_m, 'recall_std': rec_s,
        'f1_mean': f1_m, 'f1_std': f1_s
    })

    # write summary CSV
    summary_fields = ['mode','count',
                      'iou_mean','iou_std','dice_mean','dice_std',
                      'precision_mean','precision_std','recall_mean','recall_std',
                      'f1_mean','f1_std']
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # print per-model summary
    print("\n=== MODEL SUMMARY ===")
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
    print(f"Per-model summary saved -> {summary_csv}\n")

    return summary_rows

# ------------------ driver: run through models_dir and collate results ------------------
def main():
    parser = argparse.ArgumentParser(description="Run folder-mode evaluation for all .pth in a folder (three modes).")
    parser.add_argument('--models_dir', type=str, default=r"/home/nicola/Scaricati/checkpointssept17", help='Directory containing .pth checkpoint files')
    parser.add_argument('--image_dir', type=str, default=r"/home/nicola/Scaricati/smoke100kprocessed/test/images", help='Directory with images to evaluate')
    parser.add_argument('--mask_dir', type=str, default=r"/home/nicola/Scaricati/smoke100kprocessed/test/masks", help='Directory with ground-truth masks matching images')
    parser.add_argument('--out_dir', type=str, default=r"/home/nicola/Scaricati/checkpointssept17/measure_models", help='Base output directory (per-checkpoint subfolders will be created)')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--batch', action='store_true', help='Use batch mode for folder testing (faster)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--sample_n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--save_probmaps', action='store_true', help='Save probability maps (.npy) for each image')
    parser.add_argument('--model_pattern', type=str, default="*.pth")
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    models_dir = Path(args.models_dir)
    assert models_dir.exists() and models_dir.is_dir(), f"models_dir not found: {models_dir}"
    model_files = sorted(models_dir.glob(args.model_pattern))
    if len(model_files) == 0:
        print(f"[WARN] No model files found with pattern {args.model_pattern} in {models_dir}")
        return

    # will accumulate tuples (model_name, summary_rows)
    all_models_summary: List[Tuple[str, List[Dict]]] = []

    print(f"Found {len(model_files)} model(s). Device: {device}")
    for model_path in model_files:
        ckpt_name = model_path.stem
        out_subdir = Path(args.out_dir) / ckpt_name
        print(f"\n=== Processing model: {model_path.name} -> outputs: {out_subdir} ===")
        out_subdir.mkdir(parents=True, exist_ok=True)

        try:
            model = load_model(str(model_path), device)
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_path}: {e}")
            continue

        summary_rows = mode_folder_and_return_summary(
            model=model,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            size=args.size,
            device=device,
            out_dir=str(out_subdir),
            thr=args.threshold,
            sample_n=args.sample_n,
            seed=args.seed,
            batch=args.batch,
            batch_size=args.batch_size,
            save_probmaps=args.save_probmaps
        )

        all_models_summary.append((ckpt_name, summary_rows))

    # Build global summary CSV: one row per (model, mode)
    global_csv = Path(args.out_dir) / "models_metrics_summary.csv"
    fields = ['model','mode','count',
              'iou_mean','iou_std','dice_mean','dice_std',
              'precision_mean','precision_std','recall_mean','recall_std',
              'f1_mean','f1_std']
    with open(global_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model_name, summary_rows in all_models_summary:
            for r in summary_rows:
                row = {'model': model_name, 'mode': r['mode'], 'count': r['count'],
                       'iou_mean': r['iou_mean'], 'iou_std': r['iou_std'],
                       'dice_mean': r['dice_mean'], 'dice_std': r['dice_std'],
                       'precision_mean': r['precision_mean'], 'precision_std': r['precision_std'],
                       'recall_mean': r['recall_mean'], 'recall_std': r['recall_std'],
                       'f1_mean': r['f1_mean'], 'f1_std': r['f1_std']}
                w.writerow(row)

    # Print final recap in a compact readable form
    print("\n\n=== FINAL RECAP ACROSS MODELS ===")
    for model_name, summary_rows in all_models_summary:
        print(f"\nModel: {model_name}")
        for r in summary_rows:
            print(f"  {r['mode']}: IoU {r['iou_mean']:.4f}±{r['iou_std']:.4f} | Dice {r['dice_mean']:.4f}±{r['dice_std']:.4f} | "
                  f"P {r['precision_mean']:.4f}±{r['precision_std']:.4f} | R {r['recall_mean']:.4f}±{r['recall_std']:.4f} | "
                  f"F1 {r['f1_mean']:.4f}±{r['f1_std']:.4f}")

    print(f"\nGlobal summary saved -> {global_csv}")
    print("Done.")

if __name__ == '__main__':
    main()
