#!/usr/bin/env python3
"""
pos_weight.py

Calcola e stampa su stdout il pos_weight consigliato per BCEWithLogits,
a partire dalle maschere in DATA_DIR/masks.

Esegui da PyCharm: modifica DATA_DIR / BINARY_THRESHOLD se serve e Run.
Output: JSON stampato su stdout solo.

Aggiunta: mostra progresso stampando "processed N / M (P%)" ogni report_every maschere.
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import sys
import time

EPS = 1e-8

def find_masks(masks_dir: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
    files = []
    for e in exts:
        files.extend(sorted(masks_dir.glob(e)))
    return files

def load_mask_norm(path: Path, target_size=None):
    img = Image.open(path).convert("L")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def compute_pos_weight(data_dir: str, binary_threshold: float = 0.1, max_masks: int = None,
                       resize_to: tuple = None, report_every: int = 100, verbose: bool = True):
    data_dir = Path(data_dir)
    masks_dir = data_dir / "masks"
    if not masks_dir.exists():
        raise RuntimeError(f"masks dir not found: {masks_dir}")

    masks = find_masks(masks_dir)
    if len(masks) == 0:
        raise RuntimeError(f"No mask files found in {masks_dir}")

    total_masks = len(masks)
    if max_masks is not None:
        masks = masks[:max_masks]
    processed_masks = len(masks)

    total_pos = 0
    total_pixels = 0

    t0 = time.time()
    for idx, p in enumerate(masks, start=1):
        arr = load_mask_norm(p, target_size=resize_to)
        npx = arr.size
        pos = int((arr > binary_threshold).sum())
        total_pos += pos
        total_pixels += npx

        # report progress
        if verbose and (idx % report_every == 0 or idx == processed_masks):
            elapsed = time.time() - t0
            pct = (idx / processed_masks) * 100.0
            per_img = elapsed / idx
            remaining = per_img * (processed_masks - idx)
            # print simple progress line (no extra newlines)
            print(f"[{idx}/{processed_masks}] {pct:.1f}% - elapsed {elapsed:.1f}s - est remaining {remaining:.1f}s", flush=True)

    total_neg = total_pixels - total_pos
    pos_weight = float(total_neg) / (float(total_pos) + EPS)

    out = {
        "total_masks_considered": processed_masks,
        "total_pixels": int(total_pixels),
        "total_pos": int(total_pos),
        "total_neg": int(total_neg),
        "binary_threshold": float(binary_threshold),
        "pos_weight": float(pos_weight),
    }
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/nicola/Scaricati/smoke100kprocessed/train")
    p.add_argument("--binary-threshold", type=float, default=0.1)
    p.add_argument("--max-masks", type=int, default=None)
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--report-every", type=int, default=10, help="print progress every N masks (default 100)")
    p.add_argument("--no-verbose", dest="verbose", action="store_false", help="disable progress printing")
    args = p.parse_args()

    resize_to = (args.resize, args.resize) if args.resize is not None else None
    try:
        out = compute_pos_weight(args.data_dir, args.binary_threshold, args.max_masks, resize_to,
                                 report_every=args.report_every, verbose=args.verbose)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # print JSON only as final output
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
