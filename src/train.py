#!/usr/bin/env python3
"""
train.py — training loop aggiornato per maschere soft.
Modifiche:
- argparse esteso per i nuovi pesi e iperparametri della combined_loss
- passaggio degli argomenti a combined_loss in train/val
"""

import os
import csv
import time
import argparse
from datetime import datetime
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tqdm import tqdm

import torch.multiprocessing as mp

from src.dataset import SmokeDataset, get_color_transforms, get_spatial_transforms
from src.model import UNetSegmenter
from src.utils import (
    set_seed, AverageMeter, TBLogger, combined_loss,
    dice_score_proba, metrics_at_thresholds, save_checkpoint, tb_log_examples,
    iou_score_proba_soft, iou_loss_tensor
)
from config import TRAIN

# ----------------------------
# Helpers: safe save/load checkpoint + CSV/plot utilities
# ----------------------------
def safe_save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, metric=None, cfg=None):
    try:
        save_checkpoint(path, model, optimizer, scheduler, epoch, metric, cfg=cfg)
    except Exception:
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": getattr(scheduler, "state_dict", lambda: None)() if scheduler is not None else None,
            "best_metric": metric,
            "cfg": cfg
        }
        torch.save(payload, path)

def safe_load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    best_metric = None
    val_loss = None
    start_epoch = 1
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            try:
                model.load_state_dict(ckpt["model_state"])
            except Exception:
                pass
        elif "state_dict" in ckpt:
            try:
                model.load_state_dict(ckpt["state_dict"])
            except Exception:
                pass
        elif "model" in ckpt:
            try:
                model.load_state_dict(ckpt["model"])
            except Exception:
                pass
        else:
            try:
                model.load_state_dict(ckpt)
            except Exception:
                pass

        if optimizer is not None and "optim_state" in ckpt and ckpt["optim_state"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optim_state"])
            except Exception:
                pass

        if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception:
                pass

        if "epoch" in ckpt:
            try:
                start_epoch = int(ckpt["epoch"]) + 1
            except Exception:
                pass

        if "best_metric" in ckpt:
            best_metric = ckpt["best_metric"]
        if "val_loss" in ckpt:
            val_loss = ckpt["val_loss"]
        if best_metric is None and "metric" in ckpt:
            best_metric = ckpt["metric"]
    return start_epoch, best_metric, val_loss

def save_metrics_csv(log_dir, epoch_list, train_losses, val_losses, val_soft_ious, val_binary_ious, val_dices):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "metrics.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "val_loss", "val_soft_iou", "val_iou_binary", "val_dice"])
        for e, tl, vl, vsi, vib, vd in zip(epoch_list, train_losses, val_losses, val_soft_ious, val_binary_ious, val_dices):
            w.writerow([e, tl, vl, vsi, vib, vd])

def plot_metrics(log_dir, epoch_list, train_losses, val_losses, val_soft_ious, val_binary_ious, val_dices):
    os.makedirs(log_dir, exist_ok=True)
    plt.figure()
    plt.plot(epoch_list, train_losses, label="Train Loss")
    plt.plot(epoch_list, val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(log_dir, "loss_curve.png")); plt.close()

    plt.figure()
    plt.plot(epoch_list, val_soft_ious, label="Val IoU (soft)")
    plt.plot(epoch_list, val_binary_ious, label="Val IoU (binary)")
    plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(log_dir, "iou_curve.png")); plt.close()

    plt.figure()
    plt.plot(epoch_list, val_dices, label="Val Dice (proba)")
    plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(log_dir, "dice_curve.png")); plt.close()

# ----------------------------
# Argparser with new loss weights & hyperparams
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net segmenter on smoke dataset (soft masks)")
    parser.add_argument("--data_dir", type=str, default=TRAIN["data_dir"],help="Path to processed smoke data root (contains train/val/test)")
    parser.add_argument("--task", type=str, default="smoke", choices=["smoke"], help="Task (only 'smoke' supported)")
    parser.add_argument("--batch_size", type=int, default=TRAIN["batch_size"])
    parser.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    parser.add_argument("--lr", type=float, default=TRAIN["lr"])
    parser.add_argument("--checkpoint_dir", type=str, default=TRAIN["checkpoint_dir"])
    parser.add_argument("--log_dir", type=str, default=TRAIN["log_dir"])
    parser.add_argument("--img_size", type=int, default=TRAIN["img_size"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=TRAIN["patience"])
    parser.add_argument("--save_mode", choices=["best", "all"], default=TRAIN["save_mode"])
    parser.add_argument("--threshold_for_monitor", type=float, default=0.5,
                        help="Threshold for binarizing predictions when computing monitored binary IoU metric")
    parser.add_argument("--monitor_metric", choices=["soft_iou", "binary_iou"], default="soft_iou",
                        help="Which metric to use for checkpointing/early stopping. 'soft_iou' recommended when training with soft masks.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42)

    # Combined loss weights (from config defaults)
    parser.add_argument("--bce_weight", type=float, default=TRAIN.get("bce_weight", 1.0))
    parser.add_argument("--dice_weight", type=float, default=TRAIN.get("dice_weight", 1.0))
    parser.add_argument("--soft_iou_weight", type=float, default=TRAIN.get("soft_iou_weight", 0.5))
    parser.add_argument("--focal_weight", type=float, default=TRAIN.get("focal_weight", 0.5))
    parser.add_argument("--focal_gamma", type=float, default=TRAIN.get("focal_gamma", 2.0))
    parser.add_argument("--focal_alpha", type=float, default=TRAIN.get("focal_alpha", 0.25))
    parser.add_argument("--focal_tversky_weight", type=float, default=TRAIN.get("focal_tversky_weight", 0.25))
    parser.add_argument("--ft_alpha", type=float, default=TRAIN.get("ft_alpha", 0.7))
    parser.add_argument("--ft_beta", type=float, default=TRAIN.get("ft_beta", 0.3))
    parser.add_argument("--ft_gamma", type=float, default=TRAIN.get("ft_gamma", 0.75))
    parser.add_argument("--l1_weight", type=float, default=TRAIN.get("l1_weight", 0.1))
    parser.add_argument("--boundary_weight", type=float, default=TRAIN.get("boundary_weight", 0.2))

    parser.add_argument("--resume", type=str, default=r"D:\Documenti\smokedetector\smoke_best.pth", help="Path to checkpoint to resume training from")
    parser.add_argument("--save_examples_n", type=int, default=4, help="How many val examples to log to TB each epoch")
    return parser.parse_args()

# ----------------------------
# Dataloaders (unchanged)
# ----------------------------
def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    color_tf = get_color_transforms()
    spatial_factory = get_spatial_transforms

    train_img_dir = os.path.join(data_dir, "train", "images")
    train_mask_dir = os.path.join(data_dir, "train", "masks")
    val_img_dir = os.path.join(data_dir, "val", "images")
    val_mask_dir = os.path.join(data_dir, "val", "masks")

    train_ds = SmokeDataset(
        images_dir=train_img_dir,
        masks_dir=train_mask_dir,
        color_transforms=color_tf,
        spatial_transforms_factory=spatial_factory,
        target_mode="soft",
        preload=True
    )
    val_ds = SmokeDataset(
        images_dir=val_img_dir,
        masks_dir=val_mask_dir,
        color_transforms=None,
        spatial_transforms_factory=None,
        target_mode="soft",
        preload=False
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

# ----------------------------
# Training loop (pass new params to combined_loss)
# ----------------------------
def train():
    args = parse_args()
    set_seed(args.seed)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    tb_dir = os.path.join(args.log_dir, "tb", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_logger = TBLogger(log_dir=tb_dir)

    train_loader, val_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    device = torch.device(args.device)
    model = UNetSegmenter(out_channels=1, pretrained=True).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_metric = -1.0
    best_val_loss = float("inf")
    if args.resume:
        if os.path.exists(args.resume):
            print(f"-> Resuming from checkpoint: {args.resume}")
            try:
                s_epoch, s_best_metric, s_val_loss = safe_load_checkpoint(args.resume, model, optimizer, scheduler)
                if s_epoch:
                    start_epoch = s_epoch
                if s_best_metric is not None:
                    best_metric = s_best_metric
                if s_val_loss is not None:
                    best_val_loss = s_val_loss
                print(f"  Resumed. Starting epoch {start_epoch}. Best metric: {best_metric}, Best val_loss: {best_val_loss}")
            except Exception as e:
                print(f"  Warning: failed to fully resume checkpoint: {e}")

    epochs_without_improvement = 0

    epoch_list = []
    train_losses = []
    val_losses = []
    val_soft_ious = []
    val_binary_ious = []
    val_dices = []

    total_steps = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"[Train] Epoch {epoch}", unit="batch")
        for batch_idx, batch in pbar:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(
                logits, masks,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                soft_iou_weight=args.soft_iou_weight,
                focal_weight=args.focal_weight,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                focal_tversky_weight=args.focal_tversky_weight,
                ft_alpha=args.ft_alpha,
                ft_beta=args.ft_beta,
                ft_gamma=args.ft_gamma,
                l1_weight=args.l1_weight,
                boundary_weight=args.boundary_weight
            )
            loss.backward()
            optimizer.step()

            total_steps += 1
            loss_meter.update(loss.item(), n=imgs.size(0))

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "step": total_steps})

        avg_train_loss = loss_meter.avg
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss_meter = AverageMeter()
        soft_iou_meter = AverageMeter()
        binary_iou_meter = AverageMeter()
        dice_meter = AverageMeter()

        saved_probs = []
        saved_targets = []
        saved_imgs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f"[Val]   Epoch {epoch}", unit="batch"):
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                logits = model(imgs)
                loss = combined_loss(
                    logits, masks,
                    bce_weight=args.bce_weight,
                    dice_weight=args.dice_weight,
                    soft_iou_weight=args.soft_iou_weight,
                    focal_weight=args.focal_weight,
                    focal_gamma=args.focal_gamma,
                    focal_alpha=args.focal_alpha,
                    focal_tversky_weight=args.focal_tversky_weight,
                    ft_alpha=args.ft_alpha,
                    ft_beta=args.ft_beta,
                    ft_gamma=args.ft_gamma,
                    l1_weight=args.l1_weight,
                    boundary_weight=args.boundary_weight
                )
                probs = torch.sigmoid(logits)

                val_loss_meter.update(loss.item(), n=imgs.size(0))
                dice_val = dice_score_proba(probs, masks)
                dice_meter.update(dice_val, n=imgs.size(0))

                soft_iou_batch = iou_score_proba_soft(probs.detach().cpu(), masks.detach().cpu())
                soft_iou_meter.update(soft_iou_batch, n=imgs.size(0))

                metrics = metrics_at_thresholds(probs.detach().cpu(), masks.detach().cpu(), thresholds=[args.threshold_for_monitor])
                binary_iou = metrics[args.threshold_for_monitor]["iou"]
                binary_iou_meter.update(binary_iou, n=imgs.size(0))

                if len(saved_probs) < 32:
                    saved_probs.append(probs.detach().cpu())
                    saved_targets.append(masks.detach().cpu())
                    saved_imgs.append(imgs.detach().cpu())

        avg_val_loss = val_loss_meter.avg
        avg_val_soft_iou = soft_iou_meter.avg
        avg_val_binary_iou = binary_iou_meter.avg
        avg_val_dice = dice_meter.avg

        val_losses.append(avg_val_loss)
        val_soft_ious.append(avg_val_soft_iou)
        val_binary_ious.append(avg_val_binary_iou)
        val_dices.append(avg_val_dice)

        tb_logger.add_scalars("loss", {"train": avg_train_loss, "val": avg_val_loss}, epoch)
        tb_logger.add_scalars("metrics/val", {
            "soft_iou": avg_val_soft_iou,
            "iou_binary": avg_val_binary_iou,
            "dice": avg_val_dice
        }, epoch)
        tb_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar("train/total_steps", total_steps, epoch)

        # Save checkpoint by val loss
        if avg_val_loss < best_val_loss:
            prev_best_loss = best_val_loss
            best_val_loss = avg_val_loss
            safe_loss = f"{avg_val_loss:.4f}".replace('.', '_')
            val_loss_path = os.path.join(args.checkpoint_dir, f"smoke_val_loss_best_epoch{epoch:03d}_loss{safe_loss}.pth")
            safe_save_checkpoint(val_loss_path, model, optimizer, scheduler, epoch, avg_val_loss, cfg={"args": vars(args)})
            print(f"💾 New best val_loss: {avg_val_loss:.4f} (prev {prev_best_loss if prev_best_loss!=float('inf') else 'inf'}). Saved {val_loss_path}")

        if saved_probs:
            probs_to_log = torch.cat(saved_probs)[:args.save_examples_n]
            targets_to_log = torch.cat(saved_targets)[:args.save_examples_n]
            imgs_to_log = torch.cat(saved_imgs)[:args.save_examples_n]
            try:
                tb_log_examples(tb_logger, epoch, imgs_to_log, targets_to_log, probs_to_log, n=args.save_examples_n)
            except Exception:
                try:
                    tb_logger.add_image_grid("val/examples/probs", probs_to_log.repeat(1,3,1,1), epoch)
                    tb_logger.add_image_grid("val/examples/gt", targets_to_log.repeat(1,3,1,1), epoch)
                except Exception:
                    pass

        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] TrainLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | ValSoftIoU: {avg_val_soft_iou:.4f} | ValIoU@{args.threshold_for_monitor}: {avg_val_binary_iou:.4f} | ValDice(proba): {avg_val_dice:.4f} | Time: {elapsed:.1f}s")

        monitor_value = avg_val_soft_iou if args.monitor_metric == "soft_iou" else avg_val_binary_iou

        if monitor_value > best_metric:
            best_metric = monitor_value
            epochs_without_improvement = 0
            best_path = os.path.join(args.checkpoint_dir, f"smoke_best.pth")
            safe_save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_metric, cfg={"args": vars(args)})
            print(f"⭐ New best ({args.monitor_metric}): {best_metric:.4f}. Model saved to {best_path}")
        else:
            epochs_without_improvement += 1
            print(f" No improvement for {epochs_without_improvement}/{args.patience} epochs")
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

        if args.save_mode == "all":
            epoch_path = os.path.join(args.checkpoint_dir, f"smoke_epoch{epoch}.pth")
            safe_save_checkpoint(epoch_path, model, optimizer, scheduler, epoch, best_metric, cfg={"args": vars(args)})

        scheduler.step()
        epoch_list.append(epoch)

        save_metrics_csv(args.log_dir, epoch_list, train_losses, val_losses, val_soft_ious, val_binary_ious, val_dices)
        plot_metrics(args.log_dir, epoch_list, train_losses, val_losses, val_soft_ious, val_binary_ious, val_dices)

    tb_logger.close()
    print("Training finished.")

if __name__ == "__main__":
    train()