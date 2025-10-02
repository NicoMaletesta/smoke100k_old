
#!/usr/bin/env python3
"""
preprocess_folder.py

Semplice preprocess che prende una cartella di immagini e una di maschere
(e.g. immagini in "images_dir" e maschere in "masks_dir") e salva le coppie
preprocessate in out_dir/images e out_dir/masks.

Caratteristiche:
- ridimensiona immagini e maschere a dimensione quadrata (img_size)
- per le maschere usa nearest o bilinear (default: bilinear)
- se manca la maschera per un'immagine viene creata una maschera nulla (tutta 0)
- non esegue split train/val/test (solo preprocessing)
- salva metadata in metadata.csv e summary.json nella cartella di output

Esempi:
python preprocess_folder.py
python preprocess_folder.py --images_dir ./my_images --masks_dir ./my_masks --out_dir ./preprocessed --img_size 512

Il programma è comodo da lanciare anche dentro PyCharm — tutti gli argomenti hanno
valori di default e nessuno è "required".
"""

import argparse
import os
import sys
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd
import shutil


# -----------------------------
# Utility
# -----------------------------

def map_files_by_stem(folder: Path, exts=None):
    """Ritorna dict stem -> path (primo match)."""
    exts = None if exts is None else set(e.lower() for e in exts)
    d = {}
    if not folder.exists():
        return d
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if exts and p.suffix.lower().lstrip('.') not in exts:
            continue
        d[p.stem] = p
    return d


def ensure_dirs(out_dir: Path):
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)


def resize_and_save_image(src_path: Path, dst_path: Path, size: int, resample=Image.BILINEAR):
    img = Image.open(src_path).convert('RGB')
    img_resized = img.resize((size, size), resample=resample)
    img_resized.save(dst_path)


def resize_and_save_mask(src_path: Path, dst_path: Path, size: int, resample=Image.NEAREST):
    m = Image.open(src_path).convert('L')
    m_resized = m.resize((size, size), resample=resample)
    m_resized.save(dst_path)


def save_zero_mask(dst_path: Path, size: int):
    m = Image.new('L', (size, size), 0)
    m.save(dst_path)


def unique_target_path(dst_dir: Path, stem: str, ext: str = '.png') -> Path:
    """Restituisce un Path unico (se stem.png esiste aggiunge suffisso _001, _002...)."""
    p = dst_dir / (stem + ext)
    if not p.exists():
        return p
    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i:03d}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


# -----------------------------
# Main processing
# -----------------------------

def process(images_dir: Path, masks_dir: Path, out_dir: Path, img_size: int, mask_resample_mode: str, overwrite: bool, image_exts=None, mask_exts=None):
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dirs(out_dir)

    images_dir = images_dir
    masks_dir = masks_dir

    image_map = map_files_by_stem(images_dir, exts=image_exts)
    mask_map = map_files_by_stem(masks_dir, exts=mask_exts)

    if not image_map:
        print(f"ERROR: nessuna immagine trovata in {images_dir}")
        return

    # scegli tipo resample per maschere
    mask_resample = Image.BILINEAR if mask_resample_mode == 'bilinear' else Image.NEAREST

    saved = []

    for stem, img_p in tqdm(sorted(image_map.items()), desc="Processing images", unit="img"):
        out_img_path = unique_target_path(out_dir / 'images', stem, ext='.png')
        out_mask_path = unique_target_path(out_dir / 'masks', stem, ext='.png')

        try:
            resize_and_save_image(img_p, out_img_path, img_size)
            orig_mask = None
            if stem in mask_map:
                orig_mask = mask_map[stem]
                resize_and_save_mask(orig_mask, out_mask_path, img_size, resample=mask_resample)
            else:
                save_zero_mask(out_mask_path, img_size)

            saved.append({
                'img_out': str(out_img_path),
                'mask_out': str(out_mask_path),
                'orig_img': str(img_p),
                'orig_mask': str(orig_mask) if orig_mask else ''
            })
        except Exception as e:
            print(f"[WARN] failed {img_p}: {e}")

    # metadata
    md_df = pd.DataFrame(saved)
    md_path = out_dir / 'metadata.csv'
    md_df.to_csv(md_path, index=False)

    summary = {
        'n_images': len(md_df),
        'by_mask_presence': {
            'with_mask': int((md_df['orig_mask'] != '').sum()) if not md_df.empty else 0,
            'no_mask': int((md_df['orig_mask'] == '').sum()) if not md_df.empty else 0,
        }
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✅ Preprocessing completato")
    print(f"Output salvato in {out_dir}")
    print(json.dumps(summary, indent=2))


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Preprocess immagini + maschere -> coppie (out_dir/images, out_dir/masks)')
    p.add_argument('--images_dir', type=str, default=r'/home/nicola/Immagini/try dataset/images', help='Cartella contenente le immagini (default: ./images)')
    p.add_argument('--masks_dir', type=str, default=r'/home/nicola/Immagini/try dataset/masks', help='Cartella contenente le maschere (default: ./masks)')
    p.add_argument('--out_dir', type=str, default=r'/home/nicola/Immagini/manual test/train', help='Cartella di output (default: ./preprocessed)')
    p.add_argument('--img_size', type=int, default=512, help='Dimensione quadrata H=W di output (default: 512)')
    p.add_argument('--mask_resample', type=str, default='bilinear', choices=['bilinear', 'nearest'], help='Interpolazione per le maschere (default: bilinear)')
    p.add_argument('--overwrite', action='store_true', help='Se true cancella out_dir esistente')
    p.add_argument('--image_exts', nargs='*', default=['jpg','jpeg','png','bmp','tif','tiff'], help='Estensioni immagini da considerare (default common ones)')
    p.add_argument('--mask_exts', nargs='*', default=['png','bmp','tif','tiff','jpg','jpeg'], help='Estensioni maschere da considerare')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)

    process(images_dir=images_dir,
            masks_dir=masks_dir,
            out_dir=out_dir,
            img_size=int(args.img_size),
            mask_resample_mode=args.mask_resample,
            overwrite=args.overwrite,
            image_exts=args.image_exts,
            mask_exts=args.mask_exts)
