# tests/unit/test_test_smoke_masks_and_utils.py
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from src import measure_smoke

def _save_png(path: Path, arr: np.ndarray):
    Image.fromarray(arr.astype(arr.dtype)).save(path)

def test_load_mask_raw_and_soft_and_tensor_shape(tmp_path):
    arr = (np.linspace(0,255,64*64).reshape((64,64))).astype(np.uint8)
    p = tmp_path / "mask.png"
    _save_png(p, arr)
    b, s = test_smoke.load_mask_raw_and_soft(str(p), size=32)
    assert b.shape == (32,32) and s.shape == (32,32)
    t = test_smoke.to_tensor_shape(b, device=torch.device('cpu'))
    assert tuple(t.shape) == (1,1,32,32)

def test_compute_soft_precision_recall_f1_and_safe_f1():
    p = torch.zeros((1,1,4,4), dtype=torch.float32)
    g = torch.zeros((1,1,4,4), dtype=torch.float32)
    p[0,0,1:3,1:3] = 0.5
    g[0,0,1:3,1:3] = 1.0
    prec, rec, f1 = test_smoke.compute_soft_precision_recall_f1(p, g)
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0

    assert test_smoke.safe_f1_from_prec_rec(1.0, 1.0) == 1.0
    assert test_smoke.safe_f1_from_prec_rec(0.0, 0.0) == 0.0
