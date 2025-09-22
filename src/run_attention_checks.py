#!/usr/bin/env python3
"""
Run a set of deterministic checks for SEBlock / CBAM behavior and print everything to stdout.

Usage (example in PyCharm Run Config):
    python src/run_attention_checks.py --img /path/to/img.jpg --gt /path/to/gt.png --checkpoint /path/to/ckpt.pth --device cpu

All outputs printed to console so puoi incollare i risultati qui.
"""
import argparse
import sys
import numpy as np
from PIL import Image
import math
import torch
import torchvision.transforms as T

# Import model + attention from your src (assumes src is package root)
try:
    from src.model import UNetSegmenter
    from src.attention import SEBlock, CBAM
except Exception as e:
    # fallback: try importing as top-level if running from src
    try:
        from model import UNetSegmenter
        from attention import SEBlock, CBAM
    except Exception:
        print("ERROR importing UNetSegmenter / SEBlock / CBAM — check PYTHONPATH and project layout.", file=sys.stderr)
        raise

parser = argparse.ArgumentParser()
parser.add_argument("--img", default =r"/home/nicola/Scaricati/smoke100kprocessed/test/images/smoke100k-H_test_000774.png", help="Path to RGB image to run (will be resized to input-size).")
parser.add_argument("--gt", default=r"/home/nicola/Scaricati/smoke100kprocessed/test/masks/smoke100k-H_test_000774.png", help="Optional ground-truth binary mask (same image area).")
parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint (state_dict expected).")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--input-size", type=int, default=512)
parser.add_argument("--pretrained-backbone", default=True, help="Instantiate backbone with ImageNet weights if True.")
args = parser.parse_args()

device = torch.device(args.device)
print(f"[INFO] device: {device}, input-size: {args.input_size}")

# Preprocess helper
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
def preprocess_pil(pil_img, size=(512,512), device=device):
    tf = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(pil_img).unsqueeze(0).to(device)

def pil_load_rgb(path):
    im = Image.open(path).convert("RGB")
    return im

def pil_load_gray(path):
    im = Image.open(path).convert("L")
    return im

# Load image(s)
rgb_orig = pil_load_rgb(args.img)
x = preprocess_pil(rgb_orig, size=(args.input_size, args.input_size), device=device)
print(f"[INPUT] image loaded: {args.img}, resized to {args.input_size}x{args.input_size}, tensor shape {x.shape}")

gt_mask = None
if args.gt:
    gt_pil = pil_load_gray(args.gt).resize((rgb_orig.width, rgb_orig.height), Image.BILINEAR)
    gt_arr = np.array(gt_pil) > 127
    print(f"[INPUT] GT loaded: {args.gt}, shape={gt_arr.shape}, positives={int(gt_arr.sum())}")
    gt_mask = gt_arr
else:
    print("[INPUT] No GT provided; some checks will be skipped.")

# Instantiate model
print("[MODEL] Instantiating UNetSegmenter (backbone: ResNet-50 forced). pretrained_backbone:", args.pretrained_backbone)
model = UNetSegmenter(out_channels=1, pretrained=args.pretrained_backbone).to(device)
model.eval()

# Optional: load checkpoint permissive
if args.checkpoint:
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    # try removing 'module.' prefix if present
    new_sd = {}
    for k,v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    print(f"[MODEL] Loaded checkpoint (strict=False): {args.checkpoint}")

# Hook capture containers
hook_data = {
    "cbam": [],   # list of dicts per CBAM instance
    "se": []      # list of dicts per SE instance
}

# Forward hook functions
def make_cbam_hook(idx):
    def hook(module, input, output):
        # input is a tuple; in our impl we expect sa_in, s3, sk are internal, not directly available here
        # but we capture input[0] (the tensor arriving to CBAM) and output
        try:
            t_in = input[0].detach().cpu()
        except Exception:
            t_in = None
        try:
            out = output.detach().cpu()
        except Exception:
            out = None
        # save shapes and stats
        info = {"idx": idx}
        if t_in is not None:
            info["in_shape"] = tuple(t_in.shape)
            info["in_min"] = float(t_in.min()) if t_in.numel()>0 else None
            info["in_max"] = float(t_in.max()) if t_in.numel()>0 else None
            info["in_mean"] = float(t_in.mean()) if t_in.numel()>0 else None
        if out is not None:
            info["out_shape"] = tuple(out.shape)
            info["out_min"] = float(out.min()) if out.numel()>0 else None
            info["out_max"] = float(out.max()) if out.numel()>0 else None
            info["out_mean"] = float(out.mean()) if out.numel()>0 else None
        hook_data["cbam"].append(info)
    return hook

def make_se_hook(idx):
    def hook(module, input, output):
        # capture module.last_weights if exists
        info = {"idx": idx}
        try:
            lw = getattr(module, "last_weights", None)
            if lw is not None:
                arr = lw.detach().cpu().numpy()
                info["last_weights_shape"] = arr.shape
                info["last_weights_min"] = float(arr.min())
                info["last_weights_max"] = float(arr.max())
                info["last_weights_mean"] = float(arr.mean())
                info["last_weights_std"] = float(arr.std())
        except Exception as e:
            info["last_weights_error"] = str(e)
        hook_data["se"].append(info)
    return hook

# Register hooks on all CBAM and SEBlock modules found
cbam_count = 0
se_count = 0
for m in model.modules():
    if isinstance(m, CBAM):
        m.register_forward_hook(make_cbam_hook(cbam_count))
        cbam_count += 1
    if isinstance(m, SEBlock):
        m.register_forward_hook(make_se_hook(se_count))
        se_count += 1

print(f"[HOOKS] Registered {cbam_count} CBAM hook(s), {se_count} SE hook(s)")

# Run forward (no grad)
with torch.no_grad():
    out = model(x)   # forward triggers hooks
print(f"[FORWARD] model output shape: {tuple(out.shape)}, min/max/mean/std: {float(out.min()):.6f}/{float(out.max()):.6f}/{float(out.mean()):.6f}/{float(out.std()):.6f}")

# Print hook_data
print("\n=== CBAM hook summaries ===")
if len(hook_data["cbam"]) == 0:
    print("No CBAM instances captured by hooks.")
else:
    for d in hook_data["cbam"]:
        print(f"CBAM idx {d.get('idx')}: in_shape={d.get('in_shape')}, in_min={d.get('in_min')}, in_max={d.get('in_max')}, in_mean={d.get('in_mean')}; out_shape={d.get('out_shape')}, out_min={d.get('out_min')}, out_max={d.get('out_max')}, out_mean={d.get('out_mean')}")

print("\n=== SE hook summaries ===")
if len(hook_data["se"]) == 0:
    print("No SE instances captured by hooks.")
else:
    for d in hook_data["se"]:
        print(f"SE idx {d.get('idx')}: last_weights_shape={d.get('last_weights_shape')}, min/max/mean/std = {d.get('last_weights_min')}/{d.get('last_weights_max')}/{d.get('last_weights_mean')}/{d.get('last_weights_std')}")

# Additionally try to access model.attn_cbam.last_spatial and model.attn_se.last_weights if present
print("\n=== Direct module buffers (if present) ===")
if hasattr(model, "attn_cbam"):
    try:
        ls = model.attn_cbam.last_spatial
        print("model.attn_cbam.last_spatial type:", type(ls), "shape:", tuple(ls.shape) if hasattr(ls, "shape") else None)
        arr = ls.squeeze().detach().cpu().numpy()
        print("last_spatial min/max/mean/std:", float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std()))
    except Exception as e:
        print("Could not inspect model.attn_cbam.last_spatial:", e)
else:
    print("model.attn_cbam not found.")

if hasattr(model, "attn_se"):
    try:
        lw = model.attn_se.last_weights
        arr = lw.detach().cpu().numpy()
        print("model.attn_se.last_weights shape:", arr.shape, "min/max/mean/std:", float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std()))
    except Exception as e:
        print("Could not inspect model.attn_se.last_weights:", e)
else:
    print("model.attn_se not found.")

# If GT provided, compute mean inside/outside for first found spatial map (prefer last_spatial)
if args.gt:
    import cv2
    try:
        # prefer last_spatial if available
        if hasattr(model, "attn_cbam") and getattr(model.attn_cbam, "last_spatial", None) is not None:
            sp = model.attn_cbam.last_spatial.cpu().numpy().squeeze()  # HxW
        else:
            # fallback to cbam hook output (out tensor shape might be [B,1,H,W] or similar)
            found = False
            for d in hook_data["cbam"]:
                if d.get("out_shape") and len(d["out_shape"]) >= 4:
                    # we don't have the tensor itself here, so skip
                    found = True
                    break
            if not found:
                print("[GT CHECK] No spatial map available to compare with GT")
                sp = None
            else:
                sp = None
        if sp is not None:
            # upsample to original image size
            sp_norm = (sp - sp.min())/(sp.max()-sp.min()+1e-12)
            sp_up = cv2.resize(sp_norm, (rgb_orig.width, rgb_orig.height), interpolation=cv2.INTER_LINEAR)
            mean_in = float(sp_up[gt_mask].mean()) if gt_mask.sum()>0 else None
            mean_out = float(sp_up[~gt_mask].mean())
            print(f"[GT CHECK] CBAM mean_inside={mean_in}, mean_outside={mean_out}, mean_diff={None if mean_in is None else mean_in-mean_out}")
    except Exception as e:
        print("Error computing GT comparison:", e)

# Ablation: compare output with attention modules replaced by Identity
import copy
model_copy = copy.deepcopy(model)
# disable if attributes exist
if hasattr(model_copy, "attn_se"):
    model_copy.attn_se = torch.nn.Identity()
if hasattr(model_copy, "attn_cbam"):
    model_copy.attn_cbam = torch.nn.Identity()

model_copy = model_copy.to(device).eval()
with torch.no_grad():
    out_noattn = model_copy(x)
# L2 diff
diff = (out - out_noattn).abs().mean().item()
print(f"\n[ABLATION] Mean absolute difference between with-attn and no-attn outputs: {diff:.6f}")

# Gradient check: small backward to see grads on attention params (enable training mode)
print("\n[GRAD CHECK] Running one training backward with random target to inspect grads for attention params...")
model_train = UNetSegmenter(out_channels=1, pretrained=args.pretrained_backbone).to(device)
if args.checkpoint:
    # reload weights permissively (avoid strict errors)
    model_train.load_state_dict(new_sd, strict=False)
opt = torch.optim.SGD(model_train.parameters(), lr=1e-3)
model_train.train()
x_train = x.clone().to(device)
target = torch.randn_like(model_train(x_train)).to(device)
opt.zero_grad()
loss_fn = torch.nn.BCEWithLogitsLoss()
pred_train = model_train(x_train)
loss = loss_fn(pred_train, target)
loss.backward()
# inspect gradients in attention modules
any_grad_found = False
for name, p in model_train.named_parameters():
    lname = name.lower()
    if ("attn" in lname) or ("se" in lname) or ("cbam" in lname):
        grad_none = p.grad is None
        grad_mean = None if p.grad is None else float(p.grad.abs().mean())
        print(f"[GRAD] {name}: grad_none={grad_none}, mean_abs_grad={grad_mean}")
        if p.grad is not None:
            any_grad_found = True
print("[GRAD CHECK] Any grad found in attention params?:", any_grad_found)

print("\n=== END OF CHECKS ===")
