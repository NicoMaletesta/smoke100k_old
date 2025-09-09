"""
stats_smoke100k.py

Utility per analizzare Smoke100k (Fase A - sanity checks).
- Scansiona le cartelle L/M/H (train/test)
- Estrae coppie image <-> mask
- Calcola statistiche per-maschera (min,max,mean,percentili, unique_count, % zero)
- Produce CSV per-image, JSON riassuntivo, istogrammi aggregati e immagini di esempio

Dipendenze: pillow, numpy, pandas, matplotlib, tqdm

Esempio:
python stats_smoke100k.py --root_dirs /data/Smoke100k-L /data/Smoke100k-M /data/Smoke100k-H \
    --out_dir ./analysis --sample_limit 1000 --examples_per_group 5
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------
# Funzioni di utilità
# -----------------------

def find_pairs_for_root(root_dir):
    """
    Cerca la struttura standard Smoke100k sotto root_dir:
    root_dir/
      train/
        smoke_free_image/
        smoke_image/
        smoke_mask/
      test/
        ...
    Ritorna dict: { 'train': [(img_path, mask_path, smoke_present), ...], 'test': [...] }
    Nota: per smoke_free_image crea entry con mask_path=None (chi processa i dati può generare maschere zero).
    """
    root = Path(root_dir)
    res = {'train': [], 'test': []}
    for split in ('train', 'test'):
        split_dir = root / split
        if not split_dir.exists():
            continue
        free_dir = split_dir / 'smoke_free_image'
        smoke_img_dir = split_dir / 'smoke_image'
        mask_dir = split_dir / 'smoke_mask'

        # map filenames -> paths for mask and smoke_image
        mask_map = {}
        if mask_dir.exists():
            for p in mask_dir.iterdir():
                if p.is_file():
                    mask_map[p.stem] = p

        smoke_map = {}
        if smoke_img_dir.exists():
            for p in smoke_img_dir.iterdir():
                if p.is_file():
                    smoke_map[p.stem] = p

        # smoke images (have masks)
        for stem, imgp in smoke_map.items():
            m = mask_map.get(stem)
            if m is None:
                # warning: missing mask
                res[split].append((str(imgp), None, True))
            else:
                res[split].append((str(imgp), str(m), True))

        # free images (no smoke) -> mask will be None; later we can generate zeros
        if free_dir.exists():
            for p in free_dir.iterdir():
                if p.is_file():
                    res[split].append((str(p), None, False))
    return res

def load_mask_as_array(mask_path):
    """
    Carica una maschera in grayscale e ritorna numpy array float (0..255 or 0..65535 possible)
    """
    im = Image.open(mask_path)
    # convert to L (grayscale) to be robust se ha alpha o canali
    im = im.convert('L')
    arr = np.array(im)
    return arr

def compute_mask_stats(arr):
    """
    Dati i pixel (np.ndarray), ritorna dict con stats:
    min, max, mean, median, std, unique_count, percent_zero, percent_nonzero, percentiles...
    """
    if arr.size == 0:
        return {}
    flat = arr.ravel()
    mn = float(flat.min())
    mx = float(flat.max())
    mean = float(flat.mean())
    median = float(np.median(flat))
    std = float(flat.std())
    unique_vals = np.unique(flat)
    unique_count = int(unique_vals.size)
    percent_zero = float((flat == 0).sum()) / flat.size
    percent_nonzero = 1.0 - percent_zero
    pcts = np.percentile(flat, [1,5,25,50,75,95,99]).tolist()
    return {
        'min': mn,
        'max': mx,
        'mean': mean,
        'median': median,
        'std': std,
        'unique_count': unique_count,
        'percent_zero': percent_zero,
        'percent_nonzero': percent_nonzero,
        'percentiles': pcts
    }

def normalize_mask_array(arr):
    """
    Convert arr to float in [0,1]. Detecta 8-bit vs 16-bit.
    """
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    # se ha valori >255 presumiamo 16-bit
    if arr.max() > 255:
        return arr.astype(np.float32) / 65535.0
    # fallback
    mx = float(arr.max()) if arr.max() != 0 else 1.0
    return arr.astype(np.float32) / mx

def save_histogram(flat_values, out_png, title="Histogram"):
    plt.figure(figsize=(6,4))
    plt.hist(flat_values, bins=256, density=True)
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_example_visual(input_path, mask_path, out_png, vmax=None):
    """
    Crea un'immagine con tre pannelli: input RGB, mask heatmap (float 0..1), mask histogram
    mask_path may be None -> heatmap of zeros
    """
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    # input
    try:
        im = Image.open(input_path).convert("RGB")
        axes[0].imshow(im)
        axes[0].set_title("input")
        axes[0].axis('off')
    except Exception as e:
        axes[0].text(0.5,0.5,str(e))
        axes[0].set_title("input error")
        axes[0].axis('off')
    # mask heatmap
    if mask_path is None:
        mask_arr = np.zeros((im.size[1], im.size[0]), dtype=np.float32)
    else:
        mask_arr = load_mask_as_array(mask_path)
        mask_arr = normalize_mask_array(mask_arr)
    im2 = axes[1].imshow(mask_arr, vmin=0, vmax=(vmax if vmax is not None else 1.0))
    axes[1].set_title("mask (heatmap)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    # histogram
    axes[2].hist(mask_arr.ravel(), bins=256)
    axes[2].set_title("mask histogram")
    axes[2].set_xlim(0,1)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# -----------------------
# Funzione principale di analisi
# -----------------------

def analyze_roots(root_dirs, out_dir, sample_limit=None, examples_per_group=3, verbose=True):
    """
    root_dirs: list of paths to Smoke100k-L, Smoke100k-M, Smoke100k-H
    out_dir: folder dove salvare i report
    sample_limit: massimo numero di immagini per gruppo (train/test per level) da processare (None = tutti)
    examples_per_group: quante immagini esempio salvare per gruppo
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = []  # (level, root)
    for p in root_dirs:
        level = Path(p).name  # es. Smoke100k-L
        groups.append((level, p))

    warnings = []
    all_group_summaries = {}

    for level, root in groups:
        if verbose: print(f"\nAnalysing {level} @ {root}")
        pairs = find_pairs_for_root(root)
        level_dir = out_dir / level
        level_dir.mkdir(exist_ok=True)
        group_summary = {}

        for split, samples in pairs.items():
            if verbose: print(f"  split: {split} -> {len(samples)} entries")
            grp_out = level_dir / split
            grp_out.mkdir(exist_ok=True)

            rows = []
            # limit samples if requested
            iterator = samples if sample_limit is None else samples[:sample_limit]
            pbar = tqdm(iterator, desc=f"{level}/{split}", unit="img")
            example_count = 0
            aggregated_values = []
            for img_path, mask_path, smoke_present in pbar:
                row = {
                    'level': level,
                    'split': split,
                    'img_path': img_path,
                    'mask_path': mask_path if mask_path is not None else '',
                    'smoke_present': bool(smoke_present)
                }
                try:
                    if mask_path is None:
                        # mask all zero (for stats treat as zeros)
                        # we do not create file; stats computed from zeros
                        arr = np.zeros((1,), dtype=np.uint8)
                        stats = {
                            'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0,
                            'std': 0.0, 'unique_count': 1, 'percent_zero': 1.0, 'percent_nonzero': 0.0,
                            'percentiles': [0.0]*7
                        }
                    else:
                        arr = load_mask_as_array(mask_path)
                        stats = compute_mask_stats(arr)
                        aggregated_values.append(arr.ravel())
                        # small check: shapes should align with image
                        try:
                            im = Image.open(img_path)
                            if (im.size[1], im.size[0]) != arr.shape:
                                warnings.append(f"SHAPE_MISMATCH: {img_path} vs {mask_path} -> img {im.size[::-1]}, mask {arr.shape}")
                        except Exception:
                            pass
                    row.update(stats)
                except Exception as e:
                    row.update({'error': str(e)})
                    warnings.append(f"ERROR processing {img_path}: {e}")

                rows.append(row)

                # save example visuals for first few
                if example_count < examples_per_group:
                    out_png = grp_out / f"example_{Path(img_path).stem}.png"
                    try:
                        save_example_visual(img_path, mask_path, str(out_png))
                        example_count += 1
                    except Exception as e:
                        warnings.append(f"Failed to save example {img_path}: {e}")

            # save per-image CSV
            df = pd.DataFrame(rows)
            per_csv = level_dir / f"{split}_per_image_stats.csv"
            df.to_csv(per_csv, index=False)

            # aggregated histogram for group
            if aggregated_values:
                # concat but be memory careful: use limited sample if heavy
                cat = np.concatenate([v if v.size <= 1_000_000 else np.random.choice(v, size=1_000_000, replace=False) for v in aggregated_values])
                hist_png = level_dir / f"{split}_histogram.png"
                save_histogram(cat, str(hist_png), title=f"{level} {split} aggregated mask histogram")

            # compute group summary stats
            if len(rows) > 0:
                numeric_cols = ['min','max','mean','median','std','unique_count','percent_zero','percent_nonzero']
                summary = {}
                for c in numeric_cols:
                    vals = [r[c] for r in rows if c in r and r[c] is not None]
                    if len(vals) > 0:
                        summary[f"{c}_mean"] = float(np.mean(vals))
                        summary[f"{c}_median"] = float(np.median(vals))
                # how many images have > 2 unique levels (proxy continuous)
                unique_counts = [r['unique_count'] for r in rows if 'unique_count' in r]
                if unique_counts:
                    pct_many_levels = float(sum(1 for v in unique_counts if v > 2) / len(unique_counts))
                    summary['pct_images_with_more_than_2_unique_levels'] = pct_many_levels
                group_summary[split] = summary
            else:
                group_summary[split] = {}

        all_group_summaries[level] = group_summary

    # save all summaries
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_group_summaries, f, indent=2)

    # save warnings
    if warnings:
        with open(out_dir / "warnings.txt", "w") as f:
            for w in warnings:
                f.write(w + "\n")

    print(f"\nAnalysis complete. Results in {out_dir}")
    return all_group_summaries

# -----------------------
# CLI
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Smoke100k Analyzer (sanity checks)")
    parser.add_argument(
        "--root_dirs",
        nargs='+',
        default=[
            "/home/nicola/Scaricati/smoke100k/smoke100k-H",
            "/home/nicola/Scaricati/smoke100k/smoke100k-L",
            "/home/nicola/Scaricati/smoke100k/smoke100k-M",
        ],
        help="Paths to Smoke100k-L Smoke100k-M Smoke100k-H (one or more). If not provided, defaults are used."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/nicola/Scaricati/outsmoke100k",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=2000,
        help="Limit per group (train/test) of samples to analyze"
    )
    parser.add_argument(
        "--examples_per_group",
        type=int,
        default=3,
        help="How many example images to save per group (train/test per level)"
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    analyze_roots(args.root_dirs, args.out_dir, sample_limit=args.sample_limit, examples_per_group=args.examples_per_group)
