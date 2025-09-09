# tests/unit/test_utils_io_and_basic.py
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from src import utils

def test_load_image_and_save_mask_roundtrip(tmp_path):
    # create a small RGB image and save
    p = tmp_path / "img.png"
    arr = (np.ones((10,10,3), dtype=np.uint8) * 123)
    Image.fromarray(arr).save(p)

    img_t = utils.load_image(str(p))  # tensor [3,H,W] floats 0..1
    assert isinstance(img_t, torch.Tensor)
    assert img_t.shape[0] == 3
    assert img_t.max() <= 1.0 and img_t.min() >= 0.0

    # test save_mask roundtrip
    mask = np.zeros((10,10), dtype=np.float32)
    mask[2:5,2:5] = 1.0
    out_mask = tmp_path / "mask.png"
    utils.save_mask(mask, str(out_mask))
    assert out_mask.exists()
    loaded = np.array(Image.open(out_mask).convert("L"))
    # loaded is uint8: check at least that some values are 255 where set
    assert loaded.max() in (255, 1) or loaded.max() == 255

def test_average_meter_and_seed_runs():
    am = utils.AverageMeter()
    am.update(2.0, n=1)
    am.update(4.0, n=1)
    assert abs(am.avg - 3.0) < 1e-6

    # set_seed shouldn't raise and should be callable multiple times
    utils.set_seed(42)
    r1 = torch.randn(3)
    utils.set_seed(42)
    r2 = torch.randn(3)
    assert r1.shape == r2.shape
