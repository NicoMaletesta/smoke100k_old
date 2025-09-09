"""
utils.py — Loss, metriche, TensorBoard helpers e checkpointing per training
Aggiornamenti principali:
- combined_loss ora combina: Soft IoU, Soft Dice, BCEWithLogits, Focal Loss, Focal-Tversky, L1, Boundary Loss
- implementazioni stabili e compatte per Focal e Focal-Tversky
- le altre utility (dice, iou soft, sobel boundary) restano presenti
"""

import os
import json
from typing import Optional, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # determinism flags (optional)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ----------------------------
# AverageMeter
# ----------------------------
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

# ----------------------------
# IO helpers
# ----------------------------
def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).float()

def save_mask(mask_array, path: str):
    if isinstance(mask_array, torch.Tensor):
        mask_array = mask_array.detach().cpu().numpy()
    mask_uint8 = (np.clip(mask_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(mask_uint8).save(path)

# ----------------------------
# Soft Dice & Soft IoU
# ----------------------------
def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if probs.dim() == 4 and probs.size(1) == 1:
        probs = probs.squeeze(1)
    if targets.dim() == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)
    B = probs.shape[0]
    probs_flat = probs.view(B, -1)
    targets_flat = targets.view(B, -1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    loss = 1.0 - dice
    return loss.mean()

def dice_score_proba(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    if probs.dim() == 4 and probs.size(1) == 1:
        probs = probs.squeeze(1)
    if targets.dim() == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)
    B = probs.shape[0]
    p_flat = probs.view(B, -1)
    t_flat = targets.view(B, -1)
    inter = (p_flat * t_flat).sum(dim=1)
    denom = p_flat.sum(dim=1) + t_flat.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.mean().item())

def iou_score_proba_soft(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    # soft IoU (continuous) computed on probabilities and soft targets
    p = probs
    t = targets
    if p.dim() == 4 and p.size(1) == 1:
        p = p.squeeze(1)
    if t.dim() == 4 and t.size(1) == 1:
        t = t.squeeze(1)
    B = p.shape[0]
    p_flat = p.view(B, -1)
    t_flat = t.view(B, -1)
    inter = (p_flat * t_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + t_flat.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())

def metrics_at_thresholds(probs: torch.Tensor, targets: torch.Tensor, thresholds: List[float] = [0.2, 0.3, 0.5]) -> Dict[float, Dict[str, float]]:
    """
    Compute IoU and Dice at multiple thresholds.
    Returns dict: {threshold: {'iou':.., 'dice':.., 'precision':.., 'recall':..}}
    """
    out = {}
    # ensure batch dim
    if probs.dim() == 3:
        probs = probs.unsqueeze(1) if probs.dim() == 3 and probs.size(1) != 1 else probs
    B = probs.shape[0]
    for th in thresholds:
        preds = (probs > th).float()
        # dice
        p_flat = preds.view(B, -1)
        t_bin = (targets > 0.5).float()
        t_flat = t_bin.view(B, -1)
        inter = (p_flat * t_flat).sum(dim=1)
        dice = (2.0 * inter + 1e-7) / (p_flat.sum(dim=1) + t_flat.sum(dim=1) + 1e-7)
        # iou
        union = p_flat.sum(dim=1) + t_flat.sum(dim=1) - inter
        iou = (inter + 1e-7) / (union + 1e-7)
        # precision / recall
        tp = inter
        fp = p_flat.sum(dim=1) - inter
        fn = t_flat.sum(dim=1) - inter
        precision = (tp + 1e-7) / (tp + fp + 1e-7)
        recall = (tp + 1e-7) / (tp + fn + 1e-7)
        out[th] = {
            "iou": float(iou.mean().item()),
            "dice": float(dice.mean().item()),
            "precision": float(precision.mean().item()),
            "recall": float(recall.mean().item())
        }
    return out

def iou_loss_tensor(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = probs
    t = targets
    if p.dim() == 4 and p.size(1) == 1:
        p = p.squeeze(1)
    if t.dim() == 4 and t.size(1) == 1:
        t = t.squeeze(1)
    B = p.shape[0]
    p_flat = p.view(B, -1)
    t_flat = t.view(B, -1)
    inter = (p_flat * t_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + t_flat.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()

# ----------------------------
# Boundary loss (Sobel proxy)
# ----------------------------
def boundary_loss_sobel(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    p = probs
    t = targets
    if p.dim() == 4 and p.size(1) == 1:
        p = p.squeeze(1)
    if t.dim() == 4 and t.size(1) == 1:
        t = t.squeeze(1)

    device = p.device
    dtype = p.dtype
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0

    p_in = p.unsqueeze(1) if p.dim() == 3 else p
    t_in = t.unsqueeze(1) if t.dim() == 3 else t

    gx_p = F.conv2d(p_in, kx, padding=1)
    gy_p = F.conv2d(p_in, ky, padding=1)
    mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-8)

    gx_t = F.conv2d(t_in, kx, padding=1)
    gy_t = F.conv2d(t_in, ky, padding=1)
    mag_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-8)

    loss = F.l1_loss(mag_p, mag_t)
    return loss

# ----------------------------
# Focal Loss (binary) stable implementation
# ----------------------------
def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7) -> torch.Tensor:
    """
    Binary focal loss using logits (numerically stable).
    Supports soft targets (targets in [0,1]).
    Implementation follows: FL = - alpha * (1-pt)^gamma * log(pt) with pt = sigmoid(logits) for positive, (1-sigmoid) for negative.
    For soft targets we compute a balanced per-pixel term.
    """
    # flatten shapes
    if logits.dim() == 4 and logits.size(1) == 1:
        logits_flat = logits.squeeze(1).view(logits.size(0), -1)
    else:
        logits_flat = logits.view(logits.size(0), -1)
    if targets.dim() == 4 and targets.size(1) == 1:
        targets_flat = targets.squeeze(1).view(targets.size(0), -1)
    else:
        targets_flat = targets.view(targets.size(0), -1)

    prob = torch.sigmoid(logits_flat)
    # pt: probability of the true class generalised for soft targets:
    pt = prob * targets_flat + (1.0 - prob) * (1.0 - targets_flat)
    # BCE per pixel
    bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction='none')
    modulator = (1.0 - pt).pow(gamma)
    # alpha weighting: apply positive class alpha and (1-alpha) to negative,
    # for soft labels blend accordingly
    alpha_factor = targets_flat * alpha + (1.0 - targets_flat) * (1.0 - alpha)
    loss = alpha_factor * modulator * bce
    # mean over batch
    return loss.mean()

# ----------------------------
# Tversky & Focal-Tversky (works with soft targets)
# ----------------------------
def tversky_index(probs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Tversky index per-batch and returns mean over batch.
    probs: probabilities in [0,1] (shape B,1,H,W or B,H,W)
    targets: soft targets in [0,1]
    """
    p = probs
    t = targets
    if p.dim() == 4 and p.size(1) == 1:
        p = p.squeeze(1)
    if t.dim() == 4 and t.size(1) == 1:
        t = t.squeeze(1)
    B = p.shape[0]
    p_flat = p.view(B, -1)
    t_flat = t.view(B, -1)
    TP = (p_flat * t_flat).sum(dim=1)
    FP = (p_flat * (1 - t_flat)).sum(dim=1)
    FN = ((1 - p_flat) * t_flat).sum(dim=1)
    tversky = (TP + eps) / (TP + alpha * FN + beta * FP + eps)
    return tversky.mean()

def focal_tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, eps: float = 1e-7) -> torch.Tensor:
    """
    Focal-Tversky loss as (1 - Tversky)^gamma
    """
    probs = torch.sigmoid(logits)
    tversky = tversky_index(probs, targets, alpha=alpha, beta=beta, eps=eps)
    loss = (1.0 - tversky).pow(gamma)
    return loss

# ----------------------------
# Combined loss that composes multiple terms with weights
# ----------------------------
def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  bce_weight: float = 1.0,
                  dice_weight: float = 1.0,
                  soft_iou_weight: float = 0.0,
                  focal_weight: float = 0.0,
                  focal_gamma: float = 2.0,
                  focal_alpha: float = 0.25,
                  focal_tversky_weight: float = 0.0,
                  ft_alpha: float = 0.7,
                  ft_beta: float = 0.3,
                  ft_gamma: float = 0.75,
                  l1_weight: float = 0.0,
                  boundary_weight: float = 0.0) -> torch.Tensor:
    """
    Weighted sum of:
      - BCEWithLogits
      - Soft Dice
      - Soft IoU (1 - mean_soft_iou)
      - Focal Loss (binary, but accepts soft targets)
      - Focal-Tversky Loss
      - L1 on probabilities
      - Boundary Sobel Loss

    All weights default to 0 except BCE/Dice if desired. Returns scalar tensor.
    """
    total = 0.0
    device = logits.device
    dtype = logits.dtype

    # BCE
    if bce_weight != 0.0:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        total = total + bce_weight * bce

    # Soft Dice
    if dice_weight != 0.0:
        dice = soft_dice_loss(logits, targets)
        total = total + dice_weight * dice

    probs = torch.sigmoid(logits)

    # Soft IoU (tensor-friendly)
    if soft_iou_weight != 0.0:
        iou_l = iou_loss_tensor(probs, targets)
        total = total + soft_iou_weight * iou_l

    # Focal Loss
    if focal_weight != 0.0:
        fl = focal_loss_with_logits(logits, targets, gamma=focal_gamma, alpha=focal_alpha)
        total = total + focal_weight * fl

    # Focal-Tversky
    if focal_tversky_weight != 0.0:
        ftl = focal_tversky_loss(logits, targets, alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma)
        total = total + focal_tversky_weight * ftl

    # L1 on probabilities
    if l1_weight != 0.0:
        l1 = F.l1_loss(probs, targets)
        total = total + l1_weight * l1

    # Boundary Loss via Sobel
    if boundary_weight != 0.0:
        bnd = boundary_loss_sobel(probs, targets)
        total = total + boundary_weight * bnd

    # If nothing added, return zero tensor on correct device/type
    if isinstance(total, float):
        return torch.tensor(0.0, device=device, dtype=dtype)
    return total

# ----------------------------
# Checkpoint helpers and TB logger (unchanged behavior)
# ----------------------------
def save_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    scheduler: Optional[object], epoch: int, best_metric: float, cfg: Optional[dict] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass
    if cfg is not None:
        payload["cfg"] = cfg
    torch.save(payload, path)

def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[object] = None, map_location=None) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            pass
    return start_epoch, best_metric

class TBLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    def add_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, float(value), step)
    def add_scalars(self, tag: str, value_dict: dict, step: int):
        self.writer.add_scalars(tag, {k: float(v) for k, v in value_dict.items()}, step)
    def add_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        if img_tensor.dtype != torch.float32:
            img_tensor = img_tensor.float()
        self.writer.add_image(tag, img_tensor, step)
    def add_image_grid(self, tag: str, batch_tensor: torch.Tensor, step: int, nrow: int = 4):
        grid = vutils.make_grid(batch_tensor, nrow=nrow, normalize=False, value_range=(0,1))
        self.writer.add_image(tag, grid, step)
    def close(self):
        self.writer.close()

def tb_log_examples(tb: TBLogger, step: int, images: torch.Tensor, targets: torch.Tensor, probs: torch.Tensor, n: int = 4):
    B = images.shape[0]
    n = min(n, B)
    imgs = images[:n]
    targs = targets[:n].repeat(1, 3, 1, 1)
    pr = probs[:n].repeat(1, 3, 1, 1)
    tb.add_image_grid("examples/input", imgs, step, nrow=n)
    tb.add_image_grid("examples/gt", targs, step, nrow=n)
    tb.add_image_grid("examples/probs", pr, step, nrow=n)