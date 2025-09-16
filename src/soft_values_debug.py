
#!/usr/bin/env python3
# soft_values_debug_pycharm.py
# Esegui direttamente da PyCharm. Stampa su stdout diagnostica compatta per ogni immagine.
# Configura le costanti qui sotto se vuoi cambiare percorso / numero immagini.

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# ------------------ CONFIG (modifica solo se necessario) ------------------
CHECKPOINT_PATH = "/home/nicola/Documenti/smokeresults cell/partially_good_smokedetector/smoke_best_1409.pth"
DATA_DIR = "/home/nicola/Scaricati/smoke100kprocessed/test"   # deve contenere images/ e masks/
MAX_IMAGES = 100                 # numero di immagini da processare (es. 100)
IMG_SIZE = 512                   # ridimensionamento square prima della forward
NORMALIZE_GT = "per_mask"        # "none" o "per_mask"
BINARY_THRESHOLD = 0.1           # per convertire gt soft -> gt binary
THRESH_FOR_MONITOR = 0.5         # soglia per pred binaria nelle metriche
USE_CUDA = True                  # True se vuoi usare GPU se disponibile
# ------------------------------------------------------------------------

def find_images(images_dir: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
    files = []
    for e in exts:
        files.extend(sorted(images_dir.glob(e)))
    return files

def load_image_rgb(path: Path, target_size=None):
    im = Image.open(path).convert("RGB")
    if target_size is not None:
        im = im.resize(target_size, Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0
    return arr  # H,W,3

def load_mask_soft(path: Path, target_size=None, normalize="none"):
    m = Image.open(path).convert("L")
    if target_size is not None:
        m = m.resize(target_size, Image.BILINEAR)
    arr = np.array(m).astype(np.float32) / 255.0
    if normalize == "per_mask":
        mx = float(arr.max())
        if mx > 1e-8 and mx < 0.9999:
            arr = arr / (mx + 1e-8)
    return arr  # H,W float32 in [0,1]

def soft_iou_per_image(probs, targets, eps=1e-7):
    p = probs.reshape(-1).astype(np.float64)
    t = targets.reshape(-1).astype(np.float64)
    inter = np.sum(p * t)
    union = np.sum(p) + np.sum(t) - inter
    return float((inter + eps) / (union + eps))

def binary_iou_per_image(probs, targets_binary, threshold=0.5, eps=1e-7):
    pbin = (probs > threshold).astype(np.uint8).reshape(-1)
    t = targets_binary.reshape(-1).astype(np.uint8)
    inter = int((pbin & t).sum())
    union = int(pbin.sum() + t.sum() - inter)
    iou = float((inter + eps) / (union + eps))
    return iou

def print_row(name, mean_gt, mean_pred, soft_iou, binary_iou, delta):
    # print compact tab-separated line
    print(f"{name}\t{mean_gt:.6f}\t{mean_pred:.6f}\t{soft_iou:.4f}\t{binary_iou:.4f}\t{delta:.4f}")

def main():
    data_dir = Path(DATA_DIR)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(f"[ERROR] images/ or masks/ not found under {DATA_DIR}", file=sys.stderr)
        return

    imgs = find_images(images_dir)
    if len(imgs) == 0:
        print(f"[ERROR] No images found in {images_dir}", file=sys.stderr)
        return
    imgs = imgs[: min(MAX_IMAGES, len(imgs))]

    device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")
    print(f"[INFO] device={device}", flush=True)

    # import model.py and find model class
    try:
        import model as model_module
    except Exception as e:
        print(f"[ERROR] cannot import model.py: {e}", file=sys.stderr)
        return

    ModelClass = None
    if hasattr(model_module, "UNetSegmenter"):
        ModelClass = getattr(model_module, "UNetSegmenter")
    else:
        # search for plausible model class
        for name in dir(model_module):
            obj = getattr(model_module, name)
            if isinstance(obj, type):
                nlow = name.lower()
                if nlow.startswith("unet") or nlow.startswith("segment") or "smoke" in nlow:
                    ModelClass = obj
                    break

    if ModelClass is None:
        print("[ERROR] could not find a model class in model.py (expected UNetSegmenter or similar)", file=sys.stderr)
        return

    # instantiate model (try defaults, then fallback to common signature)
    try:
        model = ModelClass()
    except Exception:
        try:
            model = ModelClass(in_channels=3, out_channels=1)
        except Exception as e:
            print(f"[ERROR] failed to instantiate model class: {e}", file=sys.stderr)
            return

    model = model.to(device)
    model.eval()

    # load checkpoint
    ck_path = Path(CHECKPOINT_PATH)
    if not ck_path.exists():
        print(f"[ERROR] checkpoint not found: {ck_path}", file=sys.stderr)
        return
    ck = torch.load(str(ck_path), map_location=device)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    else:
        state = ck
    try:
        model.load_state_dict(state)
        print(f"[INFO] loaded checkpoint (strict=True) {ck_path}", flush=True)
    except Exception as e:
        try:
            model.load_state_dict(state, strict=False)
            print(f"[WARN] strict load failed; loaded with strict=False. {e}", flush=True)
        except Exception as e2:
            print(f"[ERROR] failed to load checkpoint: {e2}", file=sys.stderr)
            return

    # header
    print("NAME\tMEAN_GT\tMEAN_PRED\tSOFT_IOU\tBINARY_IOU\tDELTA(binary - soft)")
    rows = []
    gt_means = []
    pred_means = []
    soft_ious = []
    binary_ious = []

    for idx, img_path in enumerate(imgs):
        name = img_path.stem
        mask_path = masks_dir / (name + ".png")
        if not mask_path.exists():
            # try .jpg
            alt = masks_dir / (name + ".jpg")
            if alt.exists():
                mask_path = alt
            else:
                print(f"[WARN] mask not found for {name}, skipping", file=sys.stderr)
                continue

        # load
        img = load_image_rgb(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        gt_soft = load_mask_soft(mask_path, target_size=(IMG_SIZE, IMG_SIZE), normalize=NORMALIZE_GT)
        gt_binary = (gt_soft > BINARY_THRESHOLD).astype(np.uint8)

        # forward
        x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device).float()
        with torch.no_grad():
            out = model(x)
            # handle dict or tensor
            logits = None
            if isinstance(out, dict):
                # common keys
                for k in ("logits","out","pred","mask"):
                    if k in out:
                        logits = out[k]
                        break
            else:
                logits = out
            if logits is None:
                # try using out directly
                logits = out

            # if model returns list/tuple take first
            if isinstance(logits, (list,tuple)):
                logits = logits[0]

            # at this point logits should be a tensor
            if not isinstance(logits, torch.Tensor):
                # fallback: try to coerce numpy
                try:
                    pred_np = np.array(logits).astype(np.float32)
                except Exception:
                    print(f"[WARN] model output for {name} not tensor-like, skipping", file=sys.stderr)
                    continue
            else:
                # shape handling
                if logits.dim() == 4 and logits.size(1) > 1:
                    # assume single-channel output is first channel
                    logits = logits[:,0:1,...]
                probs_t = torch.sigmoid(logits)
                probs_np = probs_t.squeeze().cpu().numpy()
                pred_np = probs_np.astype(np.float32)

        # ensure shape
        if pred_np.shape != gt_soft.shape:
            # try simple resize via PIL
            try:
                pimg = Image.fromarray((pred_np * 255.0).astype("uint8")).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                pred_np = (np.array(pimg).astype(np.float32) / 255.0)
            except Exception:
                pred_np = np.resize(pred_np, gt_soft.shape)

        mean_gt = float(gt_soft.mean())
        mean_pred = float(pred_np.mean())
        soft_iou = soft_iou_per_image(pred_np, gt_soft)
        bin_iou = binary_iou_per_image(pred_np, (gt_soft > BINARY_THRESHOLD).astype(np.uint8), threshold=THRESH_FOR_MONITOR)
        delta = bin_iou - soft_iou

        # print row
        print_row(name, mean_gt, mean_pred, soft_iou, bin_iou, delta)

        # accumulate for summary
        gt_means.append(mean_gt)
        pred_means.append(mean_pred)
        soft_ious.append(soft_iou)
        binary_ious.append(bin_iou)

    # summary
    if len(gt_means) == 0:
        print("[ERROR] No images processed.", file=sys.stderr)
        return

    print("")
    print("SUMMARY:")
    print(f"images_processed\t{len(gt_means)}")
    print(f"mean_gt_global\t{float(np.mean(gt_means)):.6f}")
    print(f"mean_pred_global\t{float(np.mean(pred_means)):.6f}")
    print(f"mean_soft_iou\t{float(np.mean(soft_ious)):.6f}")
    print(f"mean_binary_iou\t{float(np.mean(binary_ious)):.6f}")
    print("END")
    return

if __name__ == "__main__":
    main()
