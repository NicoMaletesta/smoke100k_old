"""
dataset.py — Dataset utilities per segmentazione fumo (production-ready)

Features:
- SmokeDataset: restituisce immagini e maschere coerenti (default: maschere soft, float [0,1])
- Opzione target_mode: 'soft' (default) o 'binary' (con threshold)
- Matching mask by stem (indipendente da estensione)
- Sorted file lists per riproducibilità
- Gestione maschere mancanti -> zero-mask + log
- Factory transforms che applicano interpolazione corretta per maschere soft vs binary
- Funzione di debug per salvare esempi con overlay/heatmap
"""

import os
import glob
import random
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# ----------------------------
# Logging / misc
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
_missing_masks_log = "missing_masks.log"

# ----------------------------
# Helper IO utilities
# ----------------------------
def pil_loader(path: str) -> Image.Image:
    """Load image from disk and convert to RGB PIL.Image."""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def find_mask_for_image(masks_dir: str, img_path: str) -> Optional[str]:
    """
    Find a mask file in masks_dir that has the same stem as img_path.
    Returns full path or None if not found.
    """
    stem = Path(img_path).stem
    # Accept any extension (png/jpg/tif/etc.)
    candidates = sorted(glob.glob(os.path.join(masks_dir, f"{stem}.*")))
    if not candidates:
        return None
    return candidates[0]  # first match (deterministic due to sorted)

def load_mask_as_float(mask_path: str) -> np.ndarray:
    """
    Load mask as numpy float32 normalized to [0,1].
    Robust to 8-bit and 16-bit grayscale images.
    """
    im = Image.open(mask_path)
    # convert to grayscale but preserve bit depth via numpy
    arr = np.array(im)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    # if 16-bit or higher
    maxv = float(arr.max()) if arr.size > 0 else 1.0
    if maxv > 255:
        return arr.astype(np.float32) / 65535.0
    # fallback
    denom = maxv if maxv > 0 else 1.0
    return arr.astype(np.float32) / denom

def save_zero_mask(dst_path: str, size: Tuple[int, int]):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    z = Image.new("L", size, 0)
    z.save(dst_path)

# ----------------------------
# Transforms factories
# ----------------------------
def get_color_transforms() -> T.Compose:
    return T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAutocontrast(p=0.5),
        T.RandomEqualize(p=0.5),
        T.GaussianBlur(kernel_size=(3,7), sigma=(0.1,2.0)),
    ])

def get_spatial_transforms(mask_soft: bool = True) -> T.Compose:
    """
    Spatial transforms to be applied jointly to (image + mask via RGBA).
    If mask_soft==True we prefer bilinear interpolation to preserve continuous values.
    If mask_soft==False we prefer nearest to avoid introducing intermediate levels.
    """
    interp = T.InterpolationMode.BILINEAR if mask_soft else T.InterpolationMode.NEAREST
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15, interpolation=interp),
        T.RandomAffine(
            degrees=0,
            translate=None,
            scale=(0.9, 1.1),
            shear=10,
            interpolation=interp
        ),
        T.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=interp),
    ])

# ----------------------------
# SmokeDataset
# ----------------------------
class SmokeDataset(Dataset):
    """
    Dataset per Smoke segmentation (processed dataset layout).
    Expects:
        images_dir/  -> all images (any extensions)
        masks_dir/   -> masks where file stem corresponds to image stem (any extension)
    Args:
        images_dir (str): path to images directory
        masks_dir  (str): path to masks directory
        color_transforms (callable|None): applied only to image (PIL)
        spatial_transforms_factory (callable): factory that receives mask_soft bool
        target_mode: 'soft' (default) or 'binary'
        binary_threshold: threshold used if target_mode == 'binary'
        preload (bool): if True, precomputes mapping and optionally checks mask existence
    Returns per item:
        dict {
            'image': tensor [3,H,W] float32 in [0,1],
            'mask' : tensor [1,H,W] float32 in [0,1] (soft or binary),
            'meta' : { 'img_path', 'mask_path' (or ''), 'orig_size': (W,H) }
        }
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        color_transforms: Optional[T.Compose] = None,
        spatial_transforms_factory = get_spatial_transforms,
        target_mode: str = "soft",
        binary_threshold: float = 0.5,
        preload: bool = True,
    ):
        assert target_mode in ("soft", "binary"), "target_mode must be 'soft' or 'binary'"
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_mode = target_mode
        self.binary_threshold = float(binary_threshold)
        self.color_transforms = color_transforms or get_color_transforms()
        self.spatial_transforms_factory = spatial_transforms_factory
        # gather sorted list for reproducibility
        self.img_paths = sorted(glob.glob(os.path.join(self.images_dir, "*")))
        if len(self.img_paths) == 0:
            logging.warning(f"No images found in {self.images_dir}")


        mask_paths_all = sorted(glob.glob(os.path.join(self.masks_dir, "*"))) if os.path.isdir(self.masks_dir) else []
        mask_map = {Path(p).stem: p for p in mask_paths_all}

        # mapping img -> mask (may be None)
        self.mapping: List[Tuple[str, Optional[str]]] = []
        for p in self.img_paths:
            stem = Path(p).stem
            m = mask_map.get(stem, None)
            self.mapping.append((p, m))

        if preload:
            self._log_missing_masks()


    def __len__(self):
        return len(self.mapping)

    def _log_missing_masks(self):
        missing = [os.path.basename(img) for img, m in self.mapping if m is None]
        if missing:
            with open(_missing_masks_log, "a") as fh:
                fh.write(f"--- Missing masks run ---\n")
                for name in missing:
                    fh.write(f"{name}\n")
            logging.warning(f"{len(missing)} images without masks logged in {_missing_masks_log}")

    def __getitem__(self, idx: int) -> Dict:
        img_path, mask_path = self.mapping[idx]
        img_pil = pil_loader(img_path)
        orig_size = img_pil.size  # (W,H)

        # load mask as numpy float [0,1] or create zeros if missing
        if mask_path is None:
            # create zero mask PIL
            mask_pil = Image.new("L", img_pil.size, 0)
        else:
            # load preserving values then convert back to PIL 'L' with scaled values 0..255
            mask_arr = load_mask_as_float(mask_path)  # float [0,1]
            # convert float [0,1] to 0..255 uint8 for PIL handling (we will re-normalize later)
            mask_u8 = (np.clip(mask_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            mask_pil = Image.fromarray(mask_u8, mode="L")

        # 1) color aug only on image
        image_aug = self.color_transforms(img_pil) if self.color_transforms is not None else img_pil

        # 2) joint spatial transforms: create RGBA where A is mask
        # 2) joint spatial transforms: create RGBA where A is mask
        mask_soft = (self.target_mode == "soft")

        # --- Robust handling of spatial_transforms_factory ---
        # Cases handled:
        #  - spatial_transforms_factory is None -> identity transform (no spatial aug)
        #  - spatial_transforms_factory is a factory function -> call it with mask_soft
        #  - spatial_transforms_factory is already a transform (callable) -> use as-is
        if self.spatial_transforms_factory is None:
            # identity transform: returns the PIL combined image unchanged
            def spatial_tf(x):
                return x
        else:
            # try to call as factory(mask_soft=...)
            try:
                spatial_tf = self.spatial_transforms_factory(mask_soft=mask_soft)
            except TypeError:
                # not a factory that accepts mask_soft: assume it's already a transform callable
                spatial_tf = self.spatial_transforms_factory        # create RGBA: R,G,B from image_aug, A from mask_pil
        # ensure same size
        if image_aug.size != mask_pil.size:
            mask_pil = mask_pil.resize(image_aug.size, resample=Image.NEAREST)
        combined = Image.merge("RGBA", [
            image_aug.split()[0],
            image_aug.split()[1],
            image_aug.split()[2],
            mask_pil
        ])
        combined = spatial_tf(combined)

        # split back
        r, g, b, a = combined.split()
        image_final = Image.merge("RGB", (r, g, b))
        mask_after = a  # PIL 'L' 0..255

        # 3) to tensor: image and mask as float [0,1]
        image_tensor = TF.to_tensor(image_final).type(torch.float32)  # [3,H,W]
        mask_tensor = TF.to_tensor(mask_after).type(torch.float32)   # [1,H,W], values in [0,1] because to_tensor divides by 255

        # If target_mode == 'binary' enforce binarization
        if self.target_mode == "binary":
            mask_tensor = (mask_tensor > self.binary_threshold).float()

        # meta info
        meta = {
            "img_path": img_path,
            "mask_path": mask_path if mask_path is not None else "",
            "orig_size": orig_size
        }

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": meta
        }

# ----------------------------
# Debug/utility functions
# ----------------------------
def overlay_mask_on_image(img: Image.Image, mask_arr: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlay mask heatmap on image. mask_arr in [0,1], shape (H,W)
    """

    h, w = mask_arr.shape
    img = img.convert("RGBA").resize((w, h))
    cmap = mpl.colormaps.get("magma")   # oppure mpl.colormaps["magma"]
    colored = (cmap(mask_arr)[:, :, :3] * 255).astype(np.uint8)
    heat = Image.fromarray(colored).convert("RGBA")
    blended = Image.blend(img, heat, alpha=alpha)
    return blended

def save_debug_samples(dataset: SmokeDataset, out_dir: str, n: int = 10, max_pixels: int = 512):
    """
    Save n examples from dataset for sanity checking:
      - input image (resized if needed)
      - GT heatmap (float)
      - overlay (image + heatmap)
    """
    os.makedirs(out_dir, exist_ok=True)
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    idxs = idxs[:min(n, len(dataset))]
    for i, idx in enumerate(idxs):
        sample = dataset[idx]
        img_t = sample["image"]  # tensor [3,H,W]
        mask_t = sample["mask"]  # tensor [1,H,W] float
        meta = sample.get("meta", {})
        img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        mask_np = mask_t.squeeze(0).cpu().numpy()
        img_pil = Image.fromarray(img_np)
        # resize for easier visualization if huge
        if max(img_pil.size) > max_pixels:
            img_pil = img_pil.resize((max_pixels, int(max_pixels * img_pil.size[1] / img_pil.size[0])))
            mask_np = np.array(Image.fromarray((mask_np * 255).astype(np.uint8)).resize(img_pil.size, resample=Image.BILINEAR)) / 255.0

        # save components
        base_name = f"sample_{i:03d}"
        img_pil.save(os.path.join(out_dir, base_name + "_image.png"))
        # save heatmap grayscale
        heat_uint8 = (np.clip(mask_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(heat_uint8).save(os.path.join(out_dir, base_name + "_mask.png"))
        # overlay
        overlay = overlay_mask_on_image(img_pil, mask_np)
        overlay.save(os.path.join(out_dir, base_name + "_overlay.png"))
        # metadata
        with open(os.path.join(out_dir, base_name + "_meta.txt"), "w") as fh:
            fh.write(str(meta))

    logging.info(f"Saved {len(idxs)} debug samples to {out_dir}")

# ----------------------------
# Test / main script (eseguibile)
# ----------------------------
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="SmokeDataset quick tester (sanity checks + debug samples)")
    parser.add_argument("--base_dir", type=str,
                        default=r"/home/nicola/Scaricati/smoke100kprocessed",
                        help="Base processed smoke dir. Default: ~/fire_smoke_segmentation/data/processed/smoke")
    parser.add_argument("--subset", type=str, default="train", help="Subset folder: train/val/test")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader quick check")
    parser.add_argument("--n_debug", type=int, default=8, help="How many debug examples to save")
    parser.add_argument("--debug_out", type=str, default=r"/home/nicola/Documenti/testsmoke100k", help="Folder where debug samples are written")
    parser.add_argument("--run_binary_check", action="store_true", help="Also run a binary-mode sanity check")
    parser.add_argument("--max_checks", type=int, default=16, help="Max number of samples to check stats for")
    args = parser.parse_args()

    # build paths
    images_dir = os.path.join(args.base_dir, args.subset, "images")
    masks_dir  = os.path.join(args.base_dir, args.subset, "masks")

    print(f"[INFO] Images dir: {images_dir}")
    print(f"[INFO] Masks  dir: {masks_dir}")

    # Quick existence checks
    if not os.path.isdir(images_dir):
        print(f"[ERROR] images_dir non trovato: {images_dir}")
        print("Esci. Controlla --base_dir e --subset oppure esegui il preprocess.")
        raise SystemExit(2)
    if not os.path.isdir(masks_dir):
        print(f"[WARNING] masks_dir non trovato: {masks_dir} (verifica se hai rinominato cartelle).")
        # non esco: il dataset può generare zero-mask, ma è buona idea avvisare
    # instantiate dataset (soft mode)
    ds = SmokeDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        color_transforms=None,                # per test rapido evitiamo jitter colori
        spatial_transforms_factory=get_spatial_transforms,
        target_mode="soft",
        binary_threshold=0.5,
        preload=True
    )

    n = len(ds)
    print(f"[INFO] Dataset loaded. Num samples: {n}")

    # sample mapping summary
    missing = sum(1 for _, m in ds.mapping if m is None)
    print(f"[INFO] Missing masks (logged): {missing} out of {n}")

    # Basic per-sample checks (up to max_checks)
    n_check = min(args.max_checks, n)
    print(f"[INFO] Running per-sample checks on first {n_check} samples ...")
    failures = 0
    for i in range(n_check):
        s = ds[i]
        img = s["image"]
        msk = s["mask"]
        meta = s.get("meta", {})

        # checks
        ok = True
        if not isinstance(img, torch.Tensor) or img.dtype != torch.float32:
            print(f"[ERR] sample {i}: image not float32 tensor")
            ok = False
        if img.ndim != 3 or img.shape[0] != 3:
            print(f"[ERR] sample {i}: image shape unexpected {img.shape}")
            ok = False
        if not isinstance(msk, torch.Tensor) or msk.dtype != torch.float32:
            print(f"[ERR] sample {i}: mask not float32 tensor")
            ok = False
        if msk.ndim != 3 or msk.shape[0] != 1:
            print(f"[ERR] sample {i}: mask shape unexpected {msk.shape}")
            ok = False
        # range checks
        img_min, img_max = float(img.min()), float(img.max())
        m_min, m_max = float(msk.min()), float(msk.max())
        if not (0.0 - 1e-6 <= img_min <= 1.0 + 1e-6 and 0.0 - 1e-6 <= img_max <= 1.0 + 1e-6):
            print(f"[WARN] sample {i}: image range {img_min:.6f}..{img_max:.6f} (expected [0,1])")
        if not (0.0 - 1e-6 <= m_min <= 1.0 + 1e-6 and 0.0 - 1e-6 <= m_max <= 1.0 + 1e-6):
            print(f"[WARN] sample {i}: mask range {m_min:.6f}..{m_max:.6f} (expected [0,1])")
        if "img_path" not in meta:
            print(f"[ERR] sample {i}: meta missing img_path")
            ok = False

        if not ok:
            failures += 1

    print(f"[INFO] Per-sample checks finished. Failures: {failures}/{n_check}")
    if failures > 0:
        print("[ERROR] Alcuni controlli non sono andati a buon fine. Verifica i messaggi precedenti.")
    else:
        print("[OK] Per-sample checks passed.")

    # DataLoader quick batch test
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    try:
        batch = next(iter(dl))
        b_img, b_msk = batch["image"], batch["mask"]
        print(f"[INFO] Batch shapes: image {b_img.shape}, mask {b_msk.shape}")
        print(f"[INFO] Batch image range: {float(b_img.min()):.6f}..{float(b_img.max()):.6f}")
        print(f"[INFO] Batch mask  range: {float(b_msk.min()):.6f}..{float(b_msk.max()):.6f}")
    except Exception as e:
        print(f"[ERROR] Failed to read a batch from DataLoader: {e}")
        raise

    # Save debug samples (overlay + heatmaps)
    debug_out = os.path.abspath(args.debug_out)
    print(f"[INFO] Saving {args.n_debug} debug samples to {debug_out} ...")
    try:
        save_debug_samples(ds, debug_out, n=args.n_debug)
    except Exception as e:
        print(f"[ERROR] save_debug_samples failed: {e}")
        raise

    # Optional: quick binary-mode comparison if requested
    if args.run_binary_check:
        print("[INFO] Running quick binary-mode dataset check (binary_threshold=0.5)...")
        ds_bin = SmokeDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            color_transforms=None,
            spatial_transforms_factory=get_spatial_transforms,
            target_mode="binary",
            binary_threshold=0.5,
            preload=False
        )
        b = ds_bin[0]
        print(f"[INFO] binary sample mask unique vals (approx): {torch.unique(b['mask']).cpu().numpy()}")
        save_debug_samples(ds_bin, os.path.join(debug_out, "binary_mode"), n=min(4, args.n_debug))

    print("[DONE] All checks complete. If all OK, dataset is ready for training.")
    # exit code: 0 if no failures in sample checks, else 3
    if failures > 0:
        raise SystemExit(3)
    else:
        raise SystemExit(0)

