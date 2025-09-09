# tests/unit/test_test_smoke_image_mode_and_overlay.py
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from src import test_smoke

class FakeModel:
    def __init__(self, prob=0.9):
        p = float(prob)
        self.logit = np.log(p / (1.0 - p))
    def to(self, device): return self
    def eval(self): return self
    def __call__(self, x):
        B, C, H, W = x.shape
        return torch.full((B,1,H,W), fill_value=self.logit, dtype=torch.float32)

def _save_image(path: Path, w=32, h=32, color=150):
    arr = np.ones((h,w,3), dtype=np.uint8) * color
    Image.fromarray(arr).save(path)

def _save_mask(path: Path, w=32, h=32):
    m = np.zeros((h,w), dtype=np.uint8)
    m[5:15,5:15] = 200
    Image.fromarray(m).save(path)

def test_mode_image_and_overlays(tmp_path):
    img = tmp_path / "img.png"
    m = tmp_path / "mask.png"
    _save_image(img)
    _save_mask(m)
    model = FakeModel(prob=0.9)
    summaries = test_smoke.mode_image(model, str(img), str(m), size=32, device=torch.device('cpu'), thr=0.5, save_probmap=True, out_dir=str(tmp_path))
    assert (tmp_path / f"overlay_{img.stem}_gt127.png").exists()
    assert (tmp_path / f"overlay_{img.stem}_gt1.png").exists()
    # prob saved as "<stem>_prob.npy" according to test_smoke implementation
    assert (tmp_path / f"{img.stem}_prob.npy").exists()
    assert isinstance(summaries, list) and len(summaries) == 3
    for s in summaries:
        assert 'mode' in s and 'iou' in s and 'precision' in s
