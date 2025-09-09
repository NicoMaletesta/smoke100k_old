#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
infer_single.py

Esegue inference pixel-wise su una singola immagine:
- Carica il modello UNet salvato in formato .pth
- Carica e pre-processa l’immagine
- Esegue forward, applica sigmoid per ottenere la probabilità di fumo/fuoco
- Salva la mappa di probabilità e un overlay sull’immagine originale
"""

import os
import argparse

import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

from src.model import UNetSegmenter
from src.utils import load_image, save_mask
import matplotlib.pyplot as plt


def load_model(model_path: str, device: str, out_channels: int = 1) -> UNetSegmenter:
    """
    Istanzia il modello, carica pesi e lo pone in modalità eval.
    """
    model = UNetSegmenter(out_channels=out_channels, pretrained=False)
    state = torch.load(model_path, map_location=device, weights_only=True )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def infer_image(
    model: UNetSegmenter,
    image_path: str,
    device: str,
    img_size: int
) -> np.ndarray:
    """
    Esegue inference su una singola immagine:
    - load_image → tensor [3,H,W] in [0,1]
    - resize a (img_size,img_size)
    - forward + sigmoid → [1,1,img_size,img_size]
    - rimappa a dimensione originale con nearest
    - ritorna maschera float in [0,1], shape [H_orig,W_orig]
    """
    # carica e normalizza
    img_tensor = load_image(image_path).to(device)           # [3,H,W]
    H, W = img_tensor.shape[1:]
    # ridimensiona per la rete
    img_resized = F.resize(img_tensor, [img_size, img_size])
    # aggiungi batch
    with torch.no_grad():
        logits = model(img_resized.unsqueeze(0))            # [1,1,sz,sz]
        probs  = torch.sigmoid(logits)                      # [1,1,sz,sz]
    probs = probs.squeeze().cpu()                           # [sz,sz]
    # to numpy
    prob_map = probs.numpy().astype(np.float32)
    # resize back a H×W
    prob_map = Image.fromarray(prob_map).resize((W, H), resample=Image.NEAREST)
    return np.array(prob_map, dtype=np.float32)


def save_probability_map(
        prob_map: np.ndarray,
        output_path: str,
        dpi: int = 100
):
    """
    Salva una heatmap colorata della probabilità con legenda:
      - Blu per 0, Rosso per 1, gradiente intermedio.
      - Usa il colormap 'bwr' e aggiunge una colorbar.
    Args:
        prob_map (np.ndarray): mappa di probabilità [H,W], float in [0,1].
        output_path (str): percorso di output (.png).
        dpi (int): risoluzione del salvataggio.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Crea figura con rapporto aspetto = mappa, aggiungi margine per colorbar
    h, w = prob_map.shape
    # proporzione colorbar: 0.05 del width
    fig_w = 6
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax.imshow(prob_map, cmap='bwr', vmin=0.0, vmax=1.0)
    ax.axis('off')  # niente assi

    # Aggiungi colorbar sotto la mappa
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Probabilità di fumo', fontsize=10)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def overlay_on_image(
    image_path: str,
    prob_map: np.ndarray,
    output_path: str,
    alpha: float = 0.5
):
    """
    Crea overlay tra immagine RGB e heatmap di probabilità.
    """
    orig = Image.open(image_path).convert("RGB")
    heat = Image.fromarray((prob_map * 255).astype(np.uint8)).convert("L")
    # color-map (rosso) via PIL
    heat_color = Image.merge("RGB", (heat, Image.new("L", heat.size, 0), Image.new("L", heat.size, 0)))
    overlay = Image.blend(orig, heat_color, alpha=alpha)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Inference pixel-wise su immagine singola"
    )
    parser.add_argument(
        "--model_path", type=str, default = r"/home/nicola/Scrivania/segmentation model/smoke/smoke_best3july.pth",
        help="Percorso al file .pth del modello (es. smoke_best.pth)"
    )
    parser.add_argument(
        "--image_path", type=str, default = r"/home/nicola/Scrivania/test image from the net/35531883543_92244425d5_o.jpg",
        help="Percorso a immagine di input (RGB)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="inference_output",
        help="Directory in cui salvare probabilità e overlay."
    )
    parser.add_argument(
        "--img_size", type=int, default=512,
        help="Dimensione H=W usata per la rete."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device per l’inference"
    )
    args = parser.parse_args()

    # 1) carica modello
    model = load_model(args.model_path, args.device)

    # 2) esegui inference
    prob_map = infer_image(
        model, args.image_path, args.device, args.img_size
    )

    # 3) salva mappa di probabilità
    prob_path = os.path.join(args.output_dir, "prob_heatmap.png")
    save_probability_map(prob_map, prob_path)
    print(f"[+] Probabilità salvate in {prob_path}")

    # 4) salva overlay
    overlay_path = os.path.join(args.output_dir, "overlay.png")
    overlay_on_image(args.image_path, prob_map, overlay_path)
    print(f"[+] Overlay salvato in {overlay_path}")


if __name__ == "__main__":
    main()
