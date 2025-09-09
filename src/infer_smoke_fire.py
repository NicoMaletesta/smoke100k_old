import os
import argparse
import cv2
import numpy as np
import logging

import torch
import torchvision.transforms.functional as F

from src.model import UNetSegmenter
from src.utils import load_image, save_mask, iou_score, dice_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
def load_model(model_path: str, device: str, out_channels: int = 1) -> UNetSegmenter:
    """
    Istanzia un modello UNetSegmenter, carica i pesi salvati e lo mette in modalità evaluation.

    Args:
        model_path (str): Percorso al file .pth del modello salvato.
        device (str): Dispositivo da usare, "cpu" o "cuda".
        out_channels (int): Numero di canali di output del modello (default 1).

    Returns:
        UNetSegmenter: Modello pronto per inferenza.
    """
    model = UNetSegmenter(out_channels=out_channels, pretrained=False).to(device)
    # weights_only=True ignora eventuali chiavi non corrispondenti
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def segment_image(
    model: UNetSegmenter,
    image_path: str,
    img_size: int,
    device: str,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Esegue l’inferenza su una singola immagine e restituisce la maschera binaria
    0/1 riportata alle dimensioni originali dell’immagine.

    Args:
        model: modello in eval mode
        image_path: percorso all’immagine
        img_size: dimensione quadrata per inferenza
        device: "cpu" o "cuda"
        threshold: soglia di binarizzazione

    Returns:
        mask_orig: maschera binaria uint8 0/1 [H,W]
    """
    # carica e normalizza in [0,1]
    img_tensor = load_image(image_path)               # [3, H, W], float
    _, orig_h, orig_w = img_tensor.shape

    # ridimensiona e aggiungi batch dim
    img_resized = F.resize(img_tensor, [img_size, img_size])
    input_batch = img_resized.unsqueeze(0).to(device)  # [1,3,img_size,img_size]

    with torch.no_grad():
        logits = model(input_batch)                    # [1,1,img_size,img_size]
        probs = torch.sigmoid(logits)
        mask_pred = (probs > threshold).float()        # [1,1,img_size,img_size]

    # to numpy binary mask
    mask_np = mask_pred.cpu().squeeze().numpy().astype(np.uint8)  # [img_size,img_size]
    # resize to original
    mask_orig = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_orig

def extract_contours(mask: np.ndarray) -> list[np.ndarray]:
    mask_255 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image: np.ndarray, contours: list[np.ndarray], color: tuple, thickness: int = 2):
    cv2.drawContours(image, contours, -1, color, thickness)


def extract_bounding_boxes(mask: np.ndarray, min_area: int = 100) -> list[tuple[int,int,int,int]]:
    """
    Estrae bounding boxes dalle componenti connesse della maschera.

    Args:
        mask: maschera binaria uint8 0/1
        min_area: area minima per accettare un box

    Returns:
        lista di (x,y,w,h)
    """
    mask_255 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            boxes.append((x, y, w, h))
    return boxes

def draw_boxes(image: np.ndarray, boxes: list, color: tuple, thickness: int = 2):
    """
    Disegna bounding boxes su un'immagine numpy.

    Args:
        image: array BGR
        boxes: lista di (x,y,w,h)
        color: BGR tuple
        thickness: spessore
    """
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

def process_image(
    img_path: str,
    out_dir: str,
    fire_model: UNetSegmenter,
    smoke_model: UNetSegmenter,
    img_size: int,
    device: str,
    min_area: int,
    threshold: float
):
    """
    Inference su singola immagine: salva maschere, immagine con box colorati
    e sovrapposizione (overlay) delle maschere sull'immagine originale.

    - Maschera fuoco in masks/fire_<basename>.png
    - Maschera fumo in masks/smoke_<basename>.png
    - Immagine con box: bboxes/<basename>.jpg
    - Overlay maschere: bboxes/<basename>_overlay.png
    """
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bboxes"), exist_ok=True)

    basename = os.path.splitext(os.path.basename(img_path))[0]

    # ===== segmentazione =====
    fire_mask = segment_image(fire_model, img_path, img_size, device, threshold)
    smoke_mask = segment_image(smoke_model, img_path, img_size, device, threshold)

    # salva maschere (0/1→0/255)
    save_mask(fire_mask, os.path.join(out_dir, "masks", f"fire_{basename}.png"))
    save_mask(smoke_mask, os.path.join(out_dir, "masks", f"smoke_{basename}.png"))

    # ===== bounding boxes =====
    fire_boxes = extract_bounding_boxes(fire_mask, min_area)
    smoke_boxes = extract_bounding_boxes(smoke_mask, min_area)

    # carica immagine originale in BGR
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Impossibile aprire {img_path}")

    # disegna i box (rosso per fuoco, blu per fumo)
    draw_boxes(img_bgr, fire_boxes, color=(0, 0, 255), thickness=2)
    draw_boxes(img_bgr, smoke_boxes, color=(255, 0, 0), thickness=2)

    # salva immagine con box
    out_img_path = os.path.join(out_dir, "bboxes", f"{basename}.jpg")
    cv2.imwrite(out_img_path, img_bgr)

    # ===== overlay maschere =====
    # carica immagine originale in RGB
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()

    # crea canali colore per le maschere
    # fuoco: verde (0,255,0), fumo: blu (0,0,255) in RGB
    fire_col = np.zeros_like(img_rgb)
    fire_col[..., 1] = 255  # verde
    smoke_col = np.zeros_like(img_rgb)
    smoke_col[..., 2] = 255  # blu

    # applichiamo le maschere con alpha
    alpha = 0.4
    overlay = np.where(fire_mask[..., None]==1,
                       (overlay*(1-alpha) + fire_col*alpha).astype(np.uint8),
                       overlay)
    overlay = np.where(smoke_mask[..., None]==1,
                       (overlay*(1-alpha) + smoke_col*alpha).astype(np.uint8),
                       overlay)

    # riconverti in BGR per salvarlo con OpenCV
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    out_overlay_path = os.path.join(out_dir, "bboxes", f"{basename}_overlay.png")
    cv2.imwrite(out_overlay_path, overlay_bgr)

    print(f"✅ Image processed: {basename}")
    print(f"   - Boxes saved to: {out_img_path}")
    print(f"   - Overlay saved to: {out_overlay_path}")


def process_video(
    video_path: str,
    out_dir: str,
    fire_model: UNetSegmenter,
    smoke_model: UNetSegmenter,
    img_size: int,
    device: str,
    min_area: int,
    threshold: float
):
    os.makedirs(os.path.join(out_dir, "video"), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, "video", f"{basename}_contours.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Avvio processing video '{basename}' – {total_frames} frame totali")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # inferenza e maschere
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        tensor = torch.from_numpy(frame_rgb.astype(np.float32)).permute(2, 0, 1)
        resized = F.resize(tensor, [img_size, img_size]).unsqueeze(0).to(device)

        with torch.no_grad():
            fire_logits = fire_model(resized)
            smoke_logits = smoke_model(resized)
            fire_mask = (torch.sigmoid(fire_logits) > threshold).float().cpu().squeeze().numpy().astype(np.uint8)
            smoke_mask = (torch.sigmoid(smoke_logits) > threshold).float().cpu().squeeze().numpy().astype(np.uint8)

        fire_mask = cv2.resize(fire_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        smoke_mask = cv2.resize(smoke_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # estrai contorni
        fire_contours = extract_contours(fire_mask)
        smoke_contours = extract_contours(smoke_mask)

        # disegna i contorni (rosso per fuoco, blu per fumo)
        draw_contours(frame, fire_contours, color=(0, 0, 255), thickness=4)
        draw_contours(frame, smoke_contours, color=(255, 0, 0), thickness=4)

        writer.write(frame)
        frame_idx += 1

        # logging ad ogni frame
        logging.info(f"Frame {frame_idx}/{total_frames} processato")

    cap.release()
    writer.release()
    logging.info(f"✅ Video salvato in '{out_path}'. Frame processati: {frame_idx}")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference fuoco e fumo su immagine o video")
    parser.add_argument("--fire_model", type=str,
                        default=r"/home/nicola/Scrivania/segmentation models/fire/fire_25june.pth",
                        help="Percorso al modello di fuoco")
    parser.add_argument("--smoke_model", type=str,
                        default=r"/home/nicola/Scrivania/segmentation models/smoke/smoke_best3july.pth",
                        help="Percorso al modello di fumo")
    parser.add_argument("--input", type=str,
                        default=r"/home/nicola/Scrivania/test image from the net/Onfire.mp4",
                        help="Percorso a immagine (.jpg/.png) o video (.mp4/.avi)")
    parser.add_argument("--output_dir", type=str,
                        default=r"/home/nicola/Scrivania/segmentation models/test combined/",
                        help="Directory di output")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Dimensione quadrata per inferenza")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositivo: cuda o cpu")
    parser.add_argument("--min_area", type=int, default=2,
                        help="Area minima per filtrare i box")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Soglia di binarizzazione")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # carica entrambi i modelli
    fire_model = load_model(args.fire_model, args.device, out_channels=1)
    smoke_model = load_model(args.smoke_model, args.device, out_channels=1)

    ext = os.path.splitext(args.input)[1].lower()
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if ext in img_exts:
        process_image(
            args.input, args.output_dir,
            fire_model, smoke_model,
            args.img_size, args.device,
            args.min_area, args.threshold
        )
    elif ext in vid_exts:
        process_video(
            args.input, args.output_dir,
            fire_model, smoke_model,
            args.img_size, args.device,
            args.min_area, args.threshold
        )
    else:
        raise ValueError(f"Estensione {ext} non supportata.")

if __name__ == "__main__":
    main()
