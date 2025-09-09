import os
import argparse
import cv2
import numpy as np
from PIL import Image
import logging
import torch
import torchvision.transforms.functional as F

from src.model import UNetSegmenter
from src.utils import load_image, save_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def load_model(model_path: str, device: str, out_channels: int = 1) -> UNetSegmenter:
    """
    Istanzia un modello UNetSegmenter, carica i pesi salvati e lo mette in modalità evaluazione.

    Args:
        model_path (str): Percorso al file .pth del modello salvato (es. "fire_best.pth").
        device (str): Dispositivo da usare, "cpu" o "cuda".
        out_channels (int): Numero di canali di output del modello (tipicamente 1).

    Returns:
        UNetSegmenter: Modello pronto per eseguire inference (model.eval()).
    """
    # Creiamo il modello senza pesi pre-trained (caricheremo i nostri)
    model = UNetSegmenter(out_channels=out_channels, pretrained=False).to(device)
    # Carichiamo lo state dict salvato
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    # Modalità evaluation: disabilita dropout, batchnorm in training, ecc.
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
    Esegue l’inferenza su una singola immagine e restituisce la maschera binaria 0/1
    alle dimensioni originali dell’immagine.

    Pipeline:
      1. Carica l’immagine con load_image (da src/utils), ottenendo un tensor [3,H,W], float 0–1.
      2. Ricava dimensioni originali (orig_h, orig_w).
      3. Ridimensiona a [3,img_size,img_size] usando F.resize.
      4. Aggiunge batch dim [1,3,img_size,img_size] e passa nel modello → logits [1,1,img_size,img_size].
      5. Applica sigmoid → probs [1,1,img_size,img_size], quindi binarizza alla soglia.
      6. Rimuove batch e canali: ottengo array [img_size,img_size] di 0/1 (uint8).
      7. Riporta la maschera alle dimensioni originali con cv2.resize (nearest neighbor).
      8. Ritorna la maschera uint8 0/1 di shape [orig_h,orig_w].

    Args:
        model (UNetSegmenter): modello in evaluation mode.
        image_path (str): percorso all’immagine RGB (.jpg/.png).
        img_size (int): dimensione quadrata a cui ridimensionare l’immagine prima di inferenza.
        device (str): "cpu" o "cuda".
        threshold (float): soglia per convertire probabilità in 0/1.

    Returns:
        mask_orig (np.ndarray): maschera binaria 0/1 alle dimensioni originali dell’immagine.
    """
    # 1) Carichiamo immagine in tensor [3,H,W], float in [0,1]
    img_tensor = load_image(image_path)  # from src/utils
    _, orig_h, orig_w = img_tensor.shape

    # 2) Ridimensioniamo a [3,img_size,img_size]
    img_resized = F.resize(img_tensor, [img_size, img_size])

    # 3) Aggiungiamo dimensione batch e spostiamo su device
    input_batch = img_resized.unsqueeze(0).to(device)  # [1,3,img_size,img_size]

    # 4) Forward nel modello
    with torch.no_grad():
        logits = model(input_batch)                 # [1,1,img_size,img_size]
        probs = torch.sigmoid(logits)               # [1,1,img_size,img_size]
        mask_pred = (probs > threshold).float()     # binarizza, rimane [1,1,img_size,img_size]

    # 5) Rimuoviamo batch e canale: -> [img_size,img_size], cast uint8
    mask_np = mask_pred.cpu().squeeze(0).squeeze(0).numpy().astype(np.uint8)

    # 6) Riportiamo la maschera alle dimensioni originali con nearest-neighbor
    mask_orig = cv2.resize(
        mask_np,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )  # risultato uint8 con valori 0/1

    return mask_orig

def segment_frame(model, frame_bgr, img_size, device, threshold=0.5):
    """
    Inferisce la maschera su un singolo frame BGR.
    """
    # BGR → RGB normalizzato in [0,1]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0
    tensor = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    # Ridimensiona per il modello
    tensor_resized = F.resize(tensor, [img_size, img_size])
    with torch.no_grad():
        logits = model(tensor_resized)           # [1,1,img_size,img_size]
        probs = torch.sigmoid(logits)[0, 0]     # [img_size,img_size]
    # Binarizza e riporta a dimensione originale
    mask_np = (probs > threshold).cpu().numpy().astype(np.uint8)
    mask_orig = cv2.resize(mask_np,
                           (frame_bgr.shape[1], frame_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return mask_orig

def extract_contours(mask):
    mask_255 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_rectangle(image, boxes, color, thickness):
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def draw_contours(image, contours, color, thickness):
    cv2.drawContours(image, contours, -1, color, thickness)


def draw_minarea_boxes(image, contours, color, thickness):
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(image, [box], 0, color, thickness)


def draw_overlay(image, mask, color, alpha):
    overlay = image.copy()
    overlay[mask == 1] = color
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)




def process_single_image(
    model, image_path, output_dir, img_size, device, min_area, threshold, draw_method
):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    mask = segment_image(model, image_path, img_size, device, threshold)
    # Salva maschera
    mask_out = os.path.join(output_dir, "masks", f"{basename}.png")
    save_mask(mask, mask_out)

    # Prepara immagine per il disegno
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Impossibile aprire {image_path}")

    # Estrai contorni e boxes
    contours = extract_contours(mask)
    boxes = [
        (x, y, w, h)
        for cnt in contours
        for x, y, w, h in [cv2.boundingRect(cnt)]
        if w * h >= min_area
    ]

    os.makedirs(os.path.join(output_dir, "bboxes"), exist_ok=True)
    out_img = os.path.join(output_dir, "bboxes", f"{basename}.jpg")

    # Dispatch del metodo di disegno
    color = (0, 0, 255)  # rosso BGR
    thickness = 2
    alpha = 0.4
    if draw_method == "rectangle":
        draw_rectangle(img, boxes, color, thickness)
    elif draw_method == "contour":
        draw_contours(img, contours, color, thickness)
    elif draw_method == "minarea":
        draw_minarea_boxes(img, contours, color, thickness)
    elif draw_method == "overlay":
        # overlay semitrasparente (utilizza mask non boxes)
        draw_overlay(img, mask, color, alpha)
    else:
        raise ValueError(f"Metodo di disegno sconosciuto: {draw_method}")

    cv2.imwrite(out_img, img)
    print(f"✅ Immagine salvata con '{draw_method}' in {out_img}")


def process_video(
    model, video_path, output_path,
    img_size, device, min_area, threshold, draw_method
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    logging.info(f"Avvio processing video: {total_frames} frame totali")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # inferenza sul singolo frame
        mask = segment_frame(model, frame_bgr, img_size, device, threshold)
        contours = extract_contours(mask)
        boxes = [
            (x, y, bw, bh)
            for cnt in contours
            for x, y, bw, bh in [cv2.boundingRect(cnt)]
            if bw * bh >= min_area
        ]

        # disegno
        color = (0, 0, 255)
        thickness = 2
        alpha = 0.4
        if draw_method == "rectangle":
            draw_rectangle(frame_bgr, boxes, color, thickness)
        elif draw_method == "contour":
            draw_contours(frame_bgr, contours, color, thickness)
        elif draw_method == "minarea":
            draw_minarea_boxes(frame_bgr, contours, color, thickness)
        elif draw_method == "overlay":
            draw_overlay(frame_bgr, mask, color, alpha)

        out_vid.write(frame_bgr)
        frame_idx += 1

        # logging ogni 10% di avanzamento
        if total_frames > 0 and frame_idx % max(1, total_frames // 10) == 0:
            pct = (frame_idx / total_frames) * 100
            logging.info(f"Processati {frame_idx}/{total_frames} frame ({pct:.1f}%)")

    cap.release()
    out_vid.release()
    logging.info(f"✅ Video salvato in '{output_path}'. Frame processati: {frame_idx}")




def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference UNetSegmenter su singola immagine o su video"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["fire", "smoke"],
        default= "smoke",
        help="Task di segmentazione: 'fire' o 'smoke'"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default = r"/home/nicola/Scrivania/segmentation model/smoke/smoke_best3july.pth",
        help="Percorso al file .pth del modello salvato (es. fire_best.pth)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default = r"/home/nicola/Scrivania/test image from the net/smoke3.mp4",
        help="Percorso a singola immagine (.jpg/.png) o file video (.mp4/.avi)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default =  r"/home/nicola/Scrivania/segmentation model/smoke/test",
        help="Directory di output dove salvare maschere e immagini/video con bounding‐box"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Dimensione (H=W) a cui ridimensionare per la inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo da usare: 'cuda' o 'cpu'"
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=2,
        help="Area minima (in pixel) per considerare una regione valida"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Soglia per convertire probabilità in binario"
    )
    parser.add_argument("--draw_method", choices=["rectangle", "contour", "minarea", "overlay"],
                        default="contour",
                        help="Metodo di disegno: rettangoli, contorni, oriented boxes o overlay")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.model_path, args.device, out_channels=1)

    ext = os.path.splitext(args.input)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if ext in image_exts:
        process_single_image(
            model, args.input, args.output_dir,
            args.img_size, args.device,
            args.min_area, args.threshold, args.draw_method
        )
    elif ext in video_exts:
        basename = os.path.splitext(os.path.basename(args.input))[0]
        out_video = os.path.join(args.output_dir, f"{basename}_{args.draw_method}.mp4")
        process_video(
            model, args.input, out_video,
            args.img_size, args.device,
            args.min_area, args.threshold, args.draw_method
        )
    else:
        raise ValueError(f"Estensione non supportata: {ext}")


if __name__ == "__main__":
    main()