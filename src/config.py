# config.py
# ─── TRAINING ────────────────────────────────────────────
TRAIN = {
    "data_dir":    r"D:\Immagini\smoke100kprocessed",
    "batch_size":   16,
    "epochs":      50,
    "img_size":   512,                # per il training
    "lr":         1e-4,

    # Loss weights

    "bce_weight": 1.0,
    "dice_weight": 1.0,
    "soft_iou_weight": 0.5,
    "focal_weight": 0.5,
    "focal_gamma": 2.0,        # gamma per focal
    "focal_alpha": 0.25,       # bilanciamento per focal (pos weight)
    "focal_tversky_weight": 0.25,
    "ft_alpha": 0.7,           # alpha per Tversky (penalizza FN più di FP)
    "ft_beta": 0.3,            # beta per Tversky
    "ft_gamma": 0.75,          # gamma per Focal-Tversky
    "l1_weight": 0.1,
    "boundary_weight": 0.2,
    "patience":   20,                 # early stopping
    "save_mode":  'best',             # 'best' | 'all'
    "checkpoint_dir": r"D:\Documenti\smokedetector8sept",
    "log_dir":        r"D:\Documenti\smokedetector8sept\logs",
    #percorso per fare il resume del training a partire da un modello ( None per partire da zero)
    "resume": None

}