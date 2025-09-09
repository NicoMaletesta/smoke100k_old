import torch
from src.model import UNetSegmenter

# 1. Carica modello
model = UNetSegmenter(out_channels=1, pretrained=False)
state_dict = torch.load(r"/home/nicola/Scrivania/segmentation model/smoke/smoke_best3july.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 2. Script del modello (no tracing)
scripted = torch.jit.script(model)

# 3. Salva per PyTorch Lite
scripted._save_for_lite_interpreter(r"/home/nicola/Scrivania/segmentation model/smoke/smoke_best3july.pt")


m = torch.jit.load(r"/home/nicola/Scrivania/segmentation model/smoke/smoke_best3july.pt", map_location="cpu")
print(type(m))  # Deve stampare: <class 'torch.jit._script.RecursiveScriptModule'>
