# tests/unit/test_utils_metrics.py
import torch
from src import utils

def make_prob_and_target(shape=(1,1,8,8), fill_prob=0.8):
    probs = torch.full(shape, fill_prob, dtype=torch.float32)
    targets = torch.zeros(shape, dtype=torch.float32)
    targets[...,2:6,2:6] = 1.0
    return probs, targets

def test_dice_and_iou_proba_soft_and_binary():
    probs, targets = make_prob_and_target()
    dice = utils.dice_score_proba(probs, targets)
    assert isinstance(dice, float)
    iou_soft = utils.iou_score_proba_soft(probs, targets)
    assert isinstance(iou_soft, float)

    # iou_score_proba thresholded and continuous
    iou_thr = utils.iou_score_proba(probs, targets, threshold=0.5)
    assert isinstance(iou_thr, float)
    iou_none = utils.iou_score_proba(probs, targets, threshold=None)
    assert isinstance(iou_none, float)

    # binary IoU on explicit binary tensors
    pred_bin = (probs > 0.5).float()
    targ_bin = (targets > 0.5).float()
    iou_bin = utils.iou_score_binary(pred_bin, targ_bin)
    assert isinstance(iou_bin, float)

def test_iou_loss_and_combined_loss_shape():
    probs, targets = make_prob_and_target()
    loss_tensor = utils.iou_loss_tensor(probs, targets)
    assert isinstance(loss_tensor, torch.Tensor)
    logits = torch.log(probs / (1 - probs + 1e-6))
    comb = utils.combined_loss(logits, targets, bce_weight=0.5, dice_weight=0.5, l1_weight=0.0)
    assert isinstance(comb, torch.Tensor)

def test_metrics_at_thresholds_keys():
    probs, targets = make_prob_and_target()
    out = utils.metrics_at_thresholds(probs, targets, thresholds=[0.3, 0.5])
    assert 0.3 in out and 0.5 in out
    for th in (0.3, 0.5):
        d = out[th]
        assert set(d.keys()) >= {"iou","dice","precision","recall"}
