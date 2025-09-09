# Fire & Smoke Segmentation

## Descrizione

Questo progetto implementa un sistema di segmentazione di immagini per rilevare fuoco e fumo in input sia statico (immagini) sia dinamico (video). Viene utilizzata una rete U-Net con backbone ResNet34 per generare maschere binarie e bounding-box sugli oggetti rilevati. È suddiviso in tre componenti principali:

1. **Preprocessing e Dataset**: script per caricare, dividere (train/val/test), ridimensionare e applicare augmentazioni (flip, rotazioni, jitter del colore) su immagini e maschere dei dataset FIRE e SMOKE.
2. **Modello**: definizione di una U-Net personalizzata (`UNetSegmenter`) che sfrutta ResNet34 come encoder.
3. **Training & Inference**: script per addestrare il modello sulle immagini preprocessate e per eseguire l’inferenza su singole immagini o video, generando maschere binarie e bounding-box.

## Requisiti

* Python ≥ 3.7
* PyTorch ≥ 1.10
* Torchvision ≥ 0.11
* NumPy
* Pillow
* OpenCV (cv2)
* scikit-learn
* Matplotlib
* tqdm



## Struttura del Progetto

```
.
├── README.md
├── data
│   ├── raw
│   │   ├── fire_kaggle        # Dataset FIRE originale (immagini e maschere)
│   │   └── smoke5k            # Dataset SMOKE originale (train/img, train/gt_, test/img, test/gt_)
│   └── processed
│       ├── fire               # Dati FIRE preprocessati (train/val/test)
│       │   ├── train
│       │   │   ├── images
│       │   │   └── masks
│       │   ├── val
│       │   │   ├── images
│       │   │   └── masks
│       │   └── test
│       │       ├── images
│       │       └── masks
│       └── smoke              # Dati SMOKE preprocessati (train/val/test analoghi)
│           ├── train
│           │   ├── images
│           │   └── masks
│           ├── val
│           │   ├── images
│           │   └── masks
│           └── test
│               ├── images
│               └── masks
├── src
│   ├── dataset.py             # Definizione dei Dataset e funzioni di preprocess
│   ├── model.py               # Definizione del modello U-Net
│   ├── utils.py               # Utility per metriche, caricamento/salvataggio immagini, logger TensorBoard
│   └── __init__.py
├── train.py                   # Script di addestramento (train U-Net su FIRE o SMOKE)
├── infer.py                   # Script di inferenza (immagine singola o video)
└── checkpoints                # Directory (creata automaticamente) in cui salvare i pesi del modello
```


## Addestramento

Lo script `train.py` gestisce l’addestramento del modello su uno dei due task (`fire` o `smoke`).

### Parametri principali:

* `--task`: `fire` o `smoke` (default: `smoke`)
* `--data_dir`: percorso alla cartella dei dati preprocessati (es. `data/processed/smoke`)
* `--batch_size`: dimensione del batch (default: 8)
* `--epochs`: numero di epoche (default: 50)
* `--lr`: learning rate iniziale (default: 1e-4)
* `--checkpoint_dir`: directory in cui salvare i pesi (`.pth`) (default: `checkpoints`)
* `--log_dir`: directory per i log di TensorBoard e metriche (default: `logs`)
* `--img_size`: dimensione di ridimensionamento per training (H=W, default: 256)
* `--device`: `cuda` o `cpu` (default: `cuda` se disponibile)
* `--patience`: epoche di early stopping (default: 15)
* `--save_mode`: `best` (salva solo il miglior modello) o `all` (salva tutti i checkpoint)

### Esempio di comando:

```bash
python train.py \
    --task fire \
    --data_dir data/processed/fire \
    --batch_size 4 \
    --epochs 30 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints/fire \
    --log_dir logs/fire \
    --img_size 256 \
    --device cuda \
    --patience 10 \
    --save_mode best
```

* Verranno salvati i pesi del miglior modello in `checkpoints/fire/fire_best.pth`.
* Durante l’addestramento, verranno generati i log TensorBoard in `logs/fire/tb/` e grafici delle metriche (`loss_curve.png`, `iou_curve.png`, `dice_curve.png`) in `logs/fire/`.
* Le metriche (per epoca) verranno anche salvate in formato CSV: `logs/fire/metrics.csv`.

## Inference

Lo script `infer.py` permette di eseguire l’inferenza sia su una singola immagine sia su un video completo.

### Parametri principali:

* `--task`: `fire` o `smoke` (utilizzato solo per etichettare, non influisce direttamente sull’inferenza)
* `--model_path`: percorso al file `.pth` del modello salvato (es. `checkpoints/fire/fire_best.pth`)
* `--input`: percorso a immagine (`.jpg`, `.png`) o video (`.mp4`, `.avi`, ecc.)
* `--output_dir`: directory di base in cui salvare risultati (es. maschere e video con bounding-box) (default: `test_results`)
* `--img_size`: dimensione di ridimensionamento per inferenza (H=W, default: 256)
* `--device`: `cuda` o `cpu` (default: `cuda` se disponibile)
* `--min_area`: area minima in pixel per considerare una regione valida (default: 10)
* `--threshold`: soglia per binarizzare la mappa di probabilità (default: 0.35)

### Inferenza su immagine:

```bash
python infer.py \
    --task smoke \
    --model_path checkpoints/smoke/smoke_best.pth \
    --input data/processed/smoke/test/images/img123.jpg \
    --output_dir results/smoke_infer \
    --img_size 256 \
    --device cuda \
    --min_area 50 \
    --threshold 0.5
```

* Vengono generati:

  * La maschera binaria (`.png`) in `results/smoke_infer/masks/img123.png`
  * L’immagine originale con bounding-box disegnati in `results/smoke_infer/bboxes/img123.jpg`
* I valori `min_area` e `threshold` possono essere regolati per filtrare piccole componenti rumorose o tarare la binarizzazione.

### Inferenza su video:

```bash
python infer.py \
    --task fire \
    --model_path checkpoints/fire/fire_best.pth \
    --input data/videos/fire_scene.mp4 \
    --output_dir results/fire_video_infer \
    --img_size 256 \
    --device cuda \
    --min_area 100 \
    --threshold 0.4
```

* Verrà prodotto un video con bounding-box dei frame in `results/fire_video_infer/fire_scene_boxed.mp4`.

## Struttura dei File Principali

### `src/utils.py`

* **`iou_score(pred, target, eps)`**: calcola l’Intersection‐over‐Union fra due tensori binari.
* **`dice_score(pred, target, eps)`**: calcola il Dice Coefficient fra due tensori binari.
* **`save_mask(mask_array, path)`**: salva una maschera binaria (0/1) come immagine PNG 0/255.
* **`load_image(path)`**: carica un’immagine RGB da disco, normalizza in \[0,1] e restituisce un tensor `[3, H, W]`.
* **`TBLogger`**: wrapper per `torch.utils.tensorboard.SummaryWriter`, comodo per salvare scalari e immagini in TensorBoard.
* **`_test_utils()`**: test rapido delle funzioni di utilità (metriche, save/load e logger).

### `src/model.py`

* **`ConvBlock(in_ch, out_ch)`**: due convoluzioni 3×3 con BatchNorm e ReLU.
* **`UpBlock(in_ch, skip_ch, out_ch)`**: upsampling bilineare + concatenazione con skip connection + `ConvBlock`.
* **`UNetSegmenter(out_channels, pretrained)`**:

  * Encoder: ResNet34 (possono essere caricati i pesi ImageNet se `pretrained=True`).
  * Bottleneck: `ConvBlock(512, 512)`.
  * Decoder: 5 livelli di `UpBlock` per ricostruire la risoluzione da 16×16 → 256×256, poi upsampling finale a 512×512.
  * Uscita: `Conv2d(64, out_channels, 1×1)` + `Upsample(scale_factor=2)`.

### `src/dataset.py`

* **`FireDataset`**:

  * Applica augmentazioni colore solo all’immagine e augmentazioni spaziali (flip/rotazione) congiunte su immagine+maschera usando canale alpha temporaneo.
  
* **`SmokeDataset`**:

  * Analoghe augmentazioni colore/spaziali su coppia immagine+maschera.
  
* **`FireProcessedDataset`**:

  * Dataset per dati “processed” di FIRE, dove immagini e maschere corrispondenti si trovano in cartelle separate ma con stesso nome di file.
  * Restituisce tensor `{"image": [3, H, W], "mask": [1, H, W]}` con maschera binaria 0/1.
* **Funzioni di supporto**:

  * `resize_and_save(src_path, dst_path, size)`: ridimensiona e salva un’immagine.
  * `generate_not_fire_masks(image_paths, output_dir, size)`: genera maschere tutte‐zero per immagini “not\_fire”.
  * `split_and_copy(...)`: divide file in split train/val/test e ridimensiona, salvando nei rispettivi output.
  * `preprocess_fire(raw_dir, out_dir, img_size)`: orchestratore per preprocessare interamente FIRE.
  * `preprocess_smoke(raw_dir, out_dir, img_size)`: orchestratore per preprocessare interamente SMOKE.

### `train.py`

* Definisce `build_dataloaders(...)` per creare DataLoader di train e val con augmentazioni su train e sola resize su val.
* Funzioni `plot_metrics(...)` e `save_metrics_csv(...)` per salvare grafici (loss, IoU, Dice) e CSV delle metriche.
* Ciclo di addestramento/validazione per ciascuna epoca:

  1. Forward pass + calcolo `BCEWithLogitsLoss` + Dice Loss
  2. Calcolo metriche IoU e Dice su validazione
  3. Early stopping basato su IoU di validazione
  4. Salvataggio checkpoint (solo “best” o ogni epoca)
  5. Salvataggio grafici e CSV in `log_dir`

### `infer.py`

* **`load_model(model_path, device, out_channels)`**: carica pesi salvati e restituisce modello in modalità `eval()`.
* **`segment_image(...)`**: esegue inferenza su singola immagine:

  1. Carica con `load_image()` tensor in \[0,1]
  2. Ridimensiona a `[3, img_size, img_size]`
  3. Forward + `sigmoid` → probabilità → binarizza con soglia
  4. Ritorna maschera binaria 0/1 riportata alle dimensioni originali con `cv2.resize`.
* **`extract_bounding_boxes(mask, min_area)`**: estrae bounding‐box delle componenti connesse usando `cv2.findContours` e filtra per area minima.
* **`draw_bounding_boxes_on_image(image_path, boxes, output_path)`**: disegna rettangoli sulla immagine originale e salva su disco.
* **`process_single_image(...)`**: inferenza su immagine singola, salva maschera e bounding‐box.
* **`process_video(...)`**: inferenza su ogni frame di un video, disegna box e salva video di output.
* `main()` definisce parsing degli argomenti e chiama la modalità corretta (immagine vs video) in base all’estensione del file.

## Esempi di Comando

1. **Preprocessing**

   ```bash
   python src/dataset.py --mode preprocess \
        --fire_raw data/raw/fire_kaggle \
        --smoke_raw data/raw/smoke5k \
        --out_dir data/processed \
        --img_size 512
   ```

2. **Addestramento (SMOKE)**

   ```bash
   python train.py \
       --task smoke \
       --data_dir data/processed/smoke \
       --batch_size 8 \
       --epochs 50 \
       --lr 1e-4 \
       --checkpoint_dir checkpoints/smoke \
       --log_dir logs/smoke \
       --img_size 256 \
       --device cuda \
       --patience 15 \
       --save_mode best
   ```

3. **Addestramento (FIRE)**

   ```bash
   python train.py \
       --task fire \
       --data_dir data/processed/fire \
       --batch_size 4 \
       --epochs 30 \
       --lr 1e-4 \
       --checkpoint_dir checkpoints/fire \
       --log_dir logs/fire \
       --img_size 256 \
       --device cuda \
       --patience 10 \
       --save_mode all
   ```

4. **Inferenza su immagine (Smoke)**

   ```bash
   python infer.py \
       --task smoke \
       --model_path checkpoints/smoke/smoke_best.pth \
       --input data/processed/smoke/test/images/img123.jpg \
       --output_dir results/smoke_infer \
       --img_size 256 \
       --device cuda \
       --min_area 50 \
       --threshold 0.5
   ```

5. **Inferenza su video (Fire)**

   ```bash
   python infer.py \
       --task fire \
       --model_path checkpoints/fire/fire_best.pth \
       --input data/videos/fire_scene.mp4 \
       --output_dir results/fire_video_infer \
       --img_size 256 \
       --device cuda \
       --min_area 100 \
       --threshold 0.4
   ```

