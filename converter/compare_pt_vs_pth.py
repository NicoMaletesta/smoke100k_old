import torch
import numpy as np
from src.model import UNetSegmenter

# --- 1) Carica il modello “da training” (.pth) ---
model_pth = UNetSegmenter(out_channels=1, pretrained=False).eval().to('cpu')
state_dict = torch.load(r"/home/nicola/Scrivania/segmentation model/smoke/smoke_best3july.pth", map_location="cpu")
model_pth.load_state_dict(state_dict)

# --- 2) Carica il modello TorchScript (.pt) ---
model_pt = torch.jit.load(r"/home/nicola/Scrivania/segmentation model/smoke/smoke_best3july.pt", map_location="cpu")
model_pt.eval()

# --- 3) Prepara un input di test con valori casuali ---
# Usiamo esattamente la stessa normalizzazione che fa TensorImageUtils in Android:
# mean/std TorchVision: [0.485,0.456,0.406], [0.229,0.224,0.225]
import torchvision.transforms as T

# Generiamo un’immagine casuale 256×256 (o 512×512 se il tuo modello .pt è stato esportato così)
H = 512
dummy_pil = torch.randint(0, 256, (3, H, H), dtype=torch.uint8)  # valori 0-255
# Convertiamo in float e normalizziamo
to_float = dummy_pil.float().div(255.0)
normalize = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
input_tensor = normalize(to_float).unsqueeze(0)  # shape [1,3,H,H]

# --- 4) Passaggio attraverso entrambi i modelli ---
with torch.no_grad():
    out_pth = torch.sigmoid(model_pth(input_tensor))   # [1,1,H,H], applichiamo sigmoid se l’output originale era un logits
    out_pt  = torch.sigmoid(model_pt(input_tensor))    # stesso shape

# --- 5) Confronto metrico ---
diff = (out_pth - out_pt).abs().max().item()
print(f"Max absolute difference tra .pth e .pt: {diff:.6f}")
if diff < 1e-5:
    print("OK: il modello .pt replica fedelmente il .pth")
else:
    print("ATTENZIONE: differenza significativa!")
