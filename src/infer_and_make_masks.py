#!/usr/bin/env python3
"""
infer_and_make_masks.py
Partendo da una cartella di immagini RGB e da un checkpoint prodotto da train.py,
per ogni immagine:
 - calcola la probability map (model -> sigmoid(logits))
 - post-process (morphology + blur) per ridurre speckle
 - genera una RGB-ground-truth dove:
     p <= t0 -> pixel nero
     p >= t1 -> pixel originale
     t0 < p < t1 -> blending lineare original * alpha (alpha = (p-t0)/(t1-t0))
 - salva:
     out_dir/probmaps/<stem>_prob.npy  (opzionale, quando --save_prob)
     out_dir/rgb_masked/<stem>.png
     out_dir/overlays/<stem>_overlay.png (diagnostico, opzionale)
Stampa progressi a schermo (tqdm + print).
"""
import os
import argparse
from pathlib import Path
import time
import random

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


from src.model import UNetSegmenter
from src.utils import load_checkpoint

def load_model(path: str, device: torch.device):
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    model = UNetSegmenter(out_channels=1, pretrained=False).to(device)
    try:
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

def preprocess_image(path: str, size: int, device: torch.device):
    img = Image.open(path).convert("RGB")
    orig = np.array(img)  # H,W,3 RGB uint8
    img_r = img.resize((size, size), resample=Image.BILINEAR)
    t = torch.from_numpy(np.array(img_r).transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)
    return t, orig

def get_logits_tensor(logits):
    """
    Normalize possibile output del modello a tensor logits Nx1xHpxWp.
    Gestisce vari formati comuni (tensor, dict{'out':...}, tuple).
    """
    if isinstance(logits, dict):
        # torchvision style
        if "out" in logits:
            return logits["out"]
        # altrimenti prendi il primo valore tensor-like
        for v in logits.values():
            if torch.is_tensor(v):
                return v
    if isinstance(logits, (list, tuple)):
        for v in logits:
            if torch.is_tensor(v):
                return v
    if torch.is_tensor(logits):
        return logits
    raise RuntimeError("Output del modello non è un tensor riconosciuto")

# ------------------ postprocess & compositing ------------------
def postprocess_prob(prob: np.ndarray, morph_k: int = 3, blur_sigma: float = 1.0):
    """
    prob: HxW float [0,1]
    ritorna HxW float [0,1] postprocessata
    """
    m = (prob * 255.0).astype(np.uint8)
    if morph_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if blur_sigma > 0:
        # GaussianBlur richiede kernel dispari; kernel 0 viene calcolato da sigma
        m = cv2.GaussianBlur(m, (0, 0), blur_sigma)
    return (m.astype(np.float32) / 255.0)

def make_rgb_masked(orig_rgb_uint8: np.ndarray, prob: np.ndarray, t0: float = 0.02, t1: float = 0.25):
    """
    orig_rgb_uint8: HxWx3 uint8 (0..255)
    prob: HxW float [0,1] (può essere di dimensione diversa: ridimensiono)
    restituisce image_uint8 HxWx3 e alpha HxW (float)
    """
    H, W = orig_rgb_uint8.shape[:2]
    if prob.shape != (H, W):
        prob_resized = cv2.resize((prob).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        prob_resized = prob
    p = np.clip(prob_resized, 0.0, 1.0)
    if t1 <= t0:
        alpha = (p >= t0).astype(np.float32)
    else:
        alpha = np.clip((p - t0) / (t1 - t0), 0.0, 1.0)
    orig_f = orig_rgb_uint8.astype(np.float32) / 255.0
    out = orig_f * alpha[:, :, None]
    out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out_u8, alpha

def save_prob_npy(prob: np.ndarray, path: str):
    np.save(path, prob.astype(np.float32))

def save_prob_png_visual(prob: np.ndarray, path: str):
    # salva una versione visiva semplice (colormap tramite cv2.applyColorMap)
    vis = (np.clip(prob,0,1)*255.0).astype(np.uint8)
    cmap = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)
    cv2.imwrite(path, cmap)

# ------------------ main flow ------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default = r"/home/nicola/Immagini/test frames", help="cartella con immagini RGB")
    p.add_argument("--out_dir", default = r"/home/nicola/Immagini/test frames/over", help="cartella di output (probmaps, rgb_masked, overlays)")
    p.add_argument("--model_path", default=r"/home/nicola/Documenti/smokedetector_2309/smoke_best.pth", help="checkpoint del modello (train.py output)")
    p.add_argument("--size", type=int, default=512, help="dimensione di input al modello (es. 512)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--t0", type=float, default=0.5, help="soglia sotto la quale pixel diventano neri")
    p.add_argument("--t1", type=float, default=0.85, help="soglia sopra la quale copio l'RGB originale")
    p.add_argument("--morph_k", type=int, default=3, help="kernel per apertura/chiusura morfologica")
    p.add_argument("--blur_sigma", type=float, default=1.0, help="sigma per Gaussian blur su probmap")
    p.add_argument("--save_prob", default= False, help="salva probmaps .npy")
    p.add_argument("--save_probvis", default= False, help="salva visualizzazione probmap .png")
    p.add_argument("--save_overlay", default= True, help="salva overlay diagnostici (alpha heatmap over image)")
    args = p.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    prob_dir = out_dir / "probmaps"
    rgb_dir = out_dir / "rgb_masked"
    overlays_dir = out_dir / "overlays"
    prob_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[INFO] device: {device}; loading model from {args.model_path} ...")
    model = load_model(args.model_path, device)
    model.to(device)
    model.eval()
    print("[INFO] model loaded. Starting inference...")

    # collect images
    img_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
    n = len(img_paths)
    if n == 0:
        print("[ERROR] nessuna immagine trovata in", image_dir)
        return

    start_time = time.time()
    # batch loop
    for i in tqdm(range(0, n, args.batch_size), desc="Batch"):
        batch_paths = img_paths[i:i+args.batch_size]
        tensors = []
        origs = []
        orig_sizes = []
        # preprocess each
        for pth in batch_paths:
            t, orig = preprocess_image(str(pth), args.size, device)
            tensors.append(t)
            origs.append(orig)
            orig_sizes.append(orig.shape[:2])
        batch_tensor = torch.cat(tensors, dim=0)  # B,C,Hp,Wp

        with torch.no_grad():
            try:
                logits = model(batch_tensor)
            except Exception as e:
                # alcuni modelli restituiscono dict o tuple; get_logits gestisce questo
                logits = model(batch_tensor)
            logits_t = get_logits_tensor(logits)  # B,1,Hp,Wp o B,Hp,Wp
            # assicurati shape Bx1xHxpW
            if logits_t.dim() == 3:
                logits_t = logits_t.unsqueeze(1)
            probs = torch.sigmoid(logits_t)  # B,1,Hp,Wp
            probs_np = probs.detach().cpu().numpy()  # B,1,Hp,Wp

        # per ogni immagine del batch: resize prob -> orig size, postprocess, save
        for j, pfile in enumerate(batch_paths):
            stem = pfile.stem
            prob_small = probs_np[j, 0]  # Hp x Wp float
            H0, W0 = orig_sizes[j]
            # resize prob to original resolution (bilinear float)
            prob_resized = cv2.resize(prob_small.astype(np.float32), (W0, H0), interpolation=cv2.INTER_LINEAR)
            # postprocess
            prob_post = postprocess_prob(prob_resized, morph_k=args.morph_k, blur_sigma=args.blur_sigma)

            # save probmap .npy if richiesto
            if args.save_prob:
                save_prob_npy(prob_post, str(prob_dir / (stem + "_prob.npy")))

            if args.save_probvis:
                save_prob_png_visual(prob_post, str(prob_dir / (stem + "_prob_vis.png")))

            # create rgb masked
            orig_rgb = origs[j]  # H,W,3 uint8
            rgb_masked, alpha = make_rgb_masked(orig_rgb, prob_post, t0=args.t0, t1=args.t1)
            cv2.imwrite(str(rgb_dir / (stem + ".png")), cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR))

            # optional overlay diag: overlay alpha as heatmap over orig
            if args.save_overlay:
                # alpha in [0,1] -> heatmap
                alpha_u8 = (np.clip(alpha,0,1)*255.0).astype(np.uint8)
                heat = cv2.applyColorMap(alpha_u8, cv2.COLORMAP_INFERNO)
                blended = cv2.addWeighted(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), 0.6, heat, 0.4, 0)
                cv2.imwrite(str(overlays_dir / (stem + "_overlay.png")), blended)

            # print brief info
            mean_p = float(prob_post.mean())
            p90 = float(np.percentile(prob_post, 90))
            print(f"[{stem}] mean_p={mean_p:.4f} p90={p90:.4f} -> saved rgb_masked")

    total = time.time() - start_time
    print(f"[DONE] processed {n} images in {total:.1f}s ({n/total:.2f} imgs/s). Outputs in {out_dir}")

if __name__ == "__main__":
    main()
