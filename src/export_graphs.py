import torch
from torch.utils.tensorboard import SummaryWriter
from src.model import UNetSegmenter  # importa il tuo modello

def export_graph(logdir=r"/home/nicola/Documenti/testsmoke100k/logs/tb/20250828-123218/graph"):
    # Inizializza modello
    model = UNetSegmenter()
    model.eval()

    # Dummy input (matcha dimensioni dataset)
    dummy_input = torch.randn(1, 3, 512, 512)

    # Writer TensorBoard
    writer = SummaryWriter(log_dir=logdir)
    writer.add_graph(model, dummy_input)
    writer.close()

    print(f"[OK] Grafo salvato in {logdir}. Avvia con:\n  tensorboard --logdir={logdir}")

if __name__ == "__main__":
    export_graph()
