#!/usr/bin/env python3
"""
preprocess_smoke100k.py

Preprocess per Smoke100k -> crea dataset ready-to-train con train/val/test
salvando nella struttura desiderata:

out_dir/
  train/
    images/
    mask/
  val/
    images/
    mask/
  test/
    images/
    mask/

Esempio:
python preprocess_smoke100k.py --root_dir /home/nicola/Scaricati/smoke100k --out_dir /home/nicola/Scaricati/smoke100k_preprocessed
"""
import os
import sys
import argparse
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import pandas as pd

# -----------------------------
# Utilità locali
# -----------------------------

def find_level_dirs(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"{root_dir} non esiste")
    levels = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and ("smoke100k" in p.name.lower()):
            levels.append(p)
    return levels

def build_mask_map(mask_dir):
    d = {}
    p = Path(mask_dir)
    if not p.exists():
        return d
    for f in p.iterdir():
        if f.is_file():
            d[f.stem] = str(f)
    return d

def resize_and_save_image(src_path, dst_path, size, resample=Image.BILINEAR):
    img = Image.open(src_path).convert("RGB")
    img_resized = img.resize((size, size), resample=resample)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img_resized.save(dst_path)

def resize_and_save_mask(src_path, dst_path, size, resample=Image.NEAREST):
    m = Image.open(src_path).convert("L")
    m_resized = m.resize((size, size), resample=resample)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    m_resized.save(dst_path)

def save_zero_mask(dst_path, size):
    m = Image.new("L", (size, size), 0)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    m.save(dst_path)

def ensure_dirs_target(out_dir):
    for subset in ("train","val","test"):
        (Path(out_dir) / subset / "images").mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / subset / "masks").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Raccolta input
# -----------------------------

def collect_inputs(level_dirs):
    train_entries, test_entries = [], []

    for lvl in level_dirs:
        lvl_name = Path(lvl).name
        for split in ("train","test"):
            split_dir = Path(lvl) / split
            if not split_dir.exists():
                continue
            smoke_img_dir = split_dir / "smoke_image"
            free_img_dir  = split_dir / "smoke_free_image"
            mask_dir      = split_dir / "smoke_mask"
            mask_map = build_mask_map(mask_dir)

            if smoke_img_dir.exists():
                for img_p in smoke_img_dir.iterdir():
                    if img_p.is_file():
                        stem = img_p.stem
                        mask_p = mask_map.get(stem, None)
                        entry = (str(img_p), str(mask_p) if mask_p else None, lvl_name, split)
                        (train_entries if split=="train" else test_entries).append(entry)

            if free_img_dir.exists():
                for img_p in free_img_dir.iterdir():
                    if img_p.is_file():
                        entry = (str(img_p), None, lvl_name, split)
                        (train_entries if split=="train" else test_entries).append(entry)

    return train_entries, test_entries

def stratified_train_val_split(train_entries, val_ratio, random_state=42):
    if val_ratio <= 0.0:
        return train_entries, []
    X = train_entries
    # stratify solo per livello (non serve smoke flag qui)
    y = [e[2] for e in train_entries]
    try:
        tr, val = train_test_split(X, test_size=val_ratio, random_state=random_state, stratify=y)
        return tr, val
    except Exception:
        random.seed(random_state)
        X_sh = X.copy()
        random.shuffle(X_sh)
        cut = int(len(X_sh)*(1.0-val_ratio))
        return X_sh[:cut], X_sh[cut:]

# -----------------------------
# Salvataggio con nomi univoci
# -----------------------------

def process_and_save(entries, out_dir, img_size, subset_name, mask_resample):
    saved = []
    counters = {}
    desc = f"Processing {subset_name}"
    for img_p, mask_p, level, split in tqdm(entries, desc=desc, unit="img"):
        # counter per livello + subset
        key = (subset_name, level)
        counters[key] = counters.get(key, 0) + 1
        counter = counters[key]

        new_name = f"{level}_{subset_name}_{counter:06d}.png"
        out_img_path = Path(out_dir) / subset_name / "images" / new_name
        out_mask_path = Path(out_dir) / subset_name / "masks" / new_name

        try:
            resize_and_save_image(img_p, str(out_img_path), img_size)
            if mask_p is None:
                save_zero_mask(str(out_mask_path), img_size)
            else:
                resize_and_save_mask(mask_p, str(out_mask_path), img_size, resample=mask_resample)

            saved.append({
                "img_out": str(out_img_path),
                "mask_out": str(out_mask_path),
                "level": level,
                "orig_img": str(img_p),
                "orig_mask": str(mask_p) if mask_p else "",
                "split": subset_name
            })
        except Exception as e:
            print(f"[WARN] failed {img_p}: {e}")
    return saved

# -----------------------------
# CLI & main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Smoke100k -> train/val/test (images + mask)")
    parser.add_argument("--root_dir", type=str, default="/home/nicola/Scaricati/smoke100k",
                        help="Cartella che contiene smoke100k-L smoke100k-M smoke100k-H")
    parser.add_argument("--out_dir", type=str, default="/home/nicola/Scaricati/smoke100kprocessed",
                        help="Cartella di output")
    parser.add_argument("--img_size", type=int, default=512, help="Dimensione quadrata H=W di output")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Frazione di validation estratta da train (0..1)")
    parser.add_argument("--mask_resample", type=str, default="bilinear", choices=["bilinear","nearest"],
                        help="Interpolazione per le maschere")
    parser.add_argument("--overwrite", action="store_true", help="Se true cancella out_dir esistente")
    parser.add_argument("--levels", nargs='*', default=None,
                        help="Lista opzionale delle cartelle level da usare")
    return parser.parse_args()

def main():
    args = parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    img_size = int(args.img_size)
    val_ratio = float(args.val_ratio)
    mask_resample = Image.BILINEAR if args.mask_resample == "bilinear" else Image.NEAREST

    if args.overwrite and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs_target(out_dir)

    level_dirs = [Path(p) for p in args.levels] if args.levels else find_level_dirs(root_dir)
    if not level_dirs:
        print(f"ERROR: non ho trovato sottocartelle smoke100k-* in {root_dir}.")
        sys.exit(1)

    print("Livelli trovati:", [str(p) for p in level_dirs])

    train_entries, test_entries = collect_inputs(level_dirs)
    print(f"Totale train candidates: {len(train_entries)}")
    print(f"Totale test candidates:  {len(test_entries)}")

    test_saved = process_and_save(test_entries, out_dir, img_size, "test", mask_resample)
    tr_entries, val_entries = stratified_train_val_split(train_entries, val_ratio, random_state=42)
    print(f"After split -> train: {len(tr_entries)}  val: {len(val_entries)}")

    train_saved = process_and_save(tr_entries, out_dir, img_size, "train", mask_resample)
    val_saved   = process_and_save(val_entries, out_dir, img_size, "val", mask_resample)

    # Salvataggio metadata e summary
    md_df = pd.DataFrame(train_saved + val_saved + test_saved)
    md_df.to_csv(Path(out_dir)/"metadata.csv", index=False)

    summary = {
        "n_total": len(md_df),
        "n_train": int((md_df['split'] == 'train').sum()),
        "n_val": int((md_df['split'] == 'val').sum()),
        "n_test": int((md_df['split'] == 'test').sum()),
        "by_level": md_df.groupby('level').size().to_dict(),
        "by_level_split": md_df.groupby(['level','split']).size().unstack(fill_value=0).to_dict(orient='index'),
    }
    with open(Path(out_dir)/"summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Preprocessing completato")
    print(f"Output salvato in {out_dir}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
