

# ===== file: config.py =====
# ─── TRAINING ────────────────────────────────────────────
TRAIN = {
    "data_dir":    r"/home/nicola/Scaricati/little_smoke100k",
    "batch_size":   16,
    "epochs":      50,
    "img_size":   512,                # per il training
    "lr":         1e-4,

    # Loss weights

    "bce_weight": 1,
    # pos_weight: float or None. If None, BCE will be called without pos_weight.
    "pos_weight": None,
    "dice_weight": 1,
    "soft_iou_weight": 0,
    "focal_weight": 0,
    "focal_gamma": 2.0,        # gamma per focal
    "focal_alpha": 0.25,       # bilanciamento per focal (pos weight)
    "focal_tversky_weight": 0,
    "ft_alpha": 0.7,           # alpha per Tversky (penalizza FN più di FP)
    "ft_beta": 0.3,            # beta per Tversky
    "ft_gamma": 0.75,          # gamma per Focal-Tversky
    "l1_weight": 0,
    "boundary_weight": 0,
    "patience":   20,                 # early stopping
    "save_mode":  'best',             # 'best' | 'all'
    "checkpoint_dir": r"/home/nicola/Scaricati/checkpointssept16",
    "log_dir":        r"/home/nicola/Scaricati/checkpointssept16/logs",
    #percorso per fare il resume del training a partire da un modello ( None per partire da zero)
    "resume": r"/home/nicola/Documenti/smokedetector_1409/smoke_val_loss_best_epoch011_loss0_5470.pth",
#r"/home/nicola/Scrivania/segmentation models/smoke/smoke_best.pth"
    # DATASET MODE
    "target_mode": "binary",          # 'soft' (default) or 'binary'
    "binary_threshold": 0.1,         # in [0,1] because dataset normalizza le maschere a [0,1]

    # monitoring defaults
    "monitor_metric": "binary_iou",  # 'soft_iou' or 'binary_iou'
    "threshold_for_monitor": 0.5,
}
