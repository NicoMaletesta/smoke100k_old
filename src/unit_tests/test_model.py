# tests/test_model.py
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from src.utils import load_image, iou_score_proba_soft as iou_score

def _make_tmp_img(path: Path, w=16, h=16, color=120):
    arr = (np.ones((h,w,3), dtype=np.uint8) * color)
    Image.fromarray(arr).save(path)

def test_load_image_basic(tmp_path):
    img_p = tmp_path / "img.png"
    _make_tmp_img(img_p)
    t = load_image(str(img_p))
    assert isinstance(t, torch.Tensor)
    # load_image returns [3,H,W]
    assert t.ndim == 3 and t.shape[0] == 3
    assert float(t.max()) <= 1.0 and float(t.min()) >= 0.0

def test_iou_score_proba_soft_simple():
    # create simple probs and targets
    probs = torch.zeros((1,1,8,8), dtype=torch.float32)
    targets = torch.zeros((1,1,8,8), dtype=torch.float32)
    probs[0,0,2:6,2:6] = 0.8
    targets[0,0,2:6,2:6] = 1.0
    iou = iou_score(probs, targets)
    assert isinstance(iou, float)
    assert 0.0 <= iou <= 1.0
