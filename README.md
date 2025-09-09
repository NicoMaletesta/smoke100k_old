# Smoke Segmentation with smoke100k dataset

## Descrizione & motivazione

Questo progetto implementa un sistema di segmentazione per rilevare **fumo** su immagini e video. L'obiettivo è produrre mappe di probabilità (soft masks) e maschere binarie che possano essere usate per allarmi automatici, analisi forense di filmati, o input a sistemi di monitoraggio. La rilevazione di fumo è particolarmente insidiosa perché il fumo può avere bassa contrasto, contorni sfumati e somiglianze cromatiche con nuvole o nebbia; per questo nel codice si usano maschere *soft*, loss composte e controlli di qualità per ridurre i falsi positivi.

---

## Panoramica del modello

Il modello principale è una U-Net con **encoder ResNet34** (opzionalmente pretrained su ImageNet) e decoder con blocchi di upsampling e doppia convoluzione.
Caratteristiche principali:

* Encoder: ResNet34 (layer1..layer4) usato come back-bone.
* Bottleneck: `ConvBlock(512,512)`.
* Decoder: più `UpBlock` che concatenano skip connections e ricostruiscono risoluzioni fino a 256×256; infine una convoluzione 1×1 e un *final upsample* per riportare l'output a 512×512 quando necessario. 
* Output: logits per canale (default `out_channels=1` per mask binaria/probabilità). :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

Per eseguire un test rapido del modello è incluso un test di sanity che costruisce un batch random e verifica le shape (vedi `src/model.py`).

---

## Principali file e cosa fanno


### `src/dataset.py`
* Implementa `SmokeDataset` che supporta due modalità di target: `soft` (default) e `binary`. Gestisce la ricerca di mask per stem, il caricamento PIL → tensor in `[0,1]`, trasformazioni colore e spaziali separate e una funzione `save_debug_samples()` per salvare overlay/heatmap degli esempi.

### `src/model.py`
* Definisce `ConvBlock`, `UpBlock` e `UNetSegmenter` con ResNet34 come encoder. Ha un upsample finale che riporta la risoluzione di output a 512×512 quando necessario. Include un piccolo test di shape.

### `src/utils.py`
* Metriche (IoU/Dice con supporto a soft targets), implementazioni di varie loss (BCEWithLogits, Soft Dice, Soft IoU, Focal, Focal-Tversky, L1 su probabilità, Boundary Sobel loss). 
* `combined_loss(...)` aggrega i termini sopra con pesi configurabili. 
* Helper per checkpointing, `TBLogger` (wrapper di `SummaryWriter`) e funzioni di logging immagini su TensorBoard. 

### `train.py`
* Loop di training completo che:
  - costruisce i dataloader (usando `SmokeDataset`),
  - crea modello `UNetSegmenter` (pretrained=True di default),
  - usa AdamW + CosineAnnealingLR,
  - calcola `combined_loss(...)` con i pesi passati via argparse o `config.py`,
  - misura metriche su validazione (soft IoU, IoU binario @ soglia, Dice su probabilità),
  - salva esempi su TensorBoard (`tb_log_examples`) e checkpoint migliori/ogni epoca. 
* Supporta resume, early stopping (patience) e salvataggio in `checkpoints/`. 

### `eval_smoke.py` (inference)
* Script di inferenza che può processare singole immagini, interi video o cartelle batch. Fornisce helper per preprocessing, conversione `probs -> mask`, salvataggio maschere e overlay RGB con bounding-box/contorni. Il caricamento checkpoint è robusto a diversi formati di payload. 

---

## Configurazione & loss predefiniti

Nel `config.py` trovi i pesi di default usati in training, tra cui i pesi della `combined_loss`. Valori di riferimento usati come default nel repo:

```json
{
  "bce_weight": 1.0,
  "dice_weight": 1.0,
  "soft_iou_weight": 0.5,
  "focal_weight": 0.5,
  "focal_gamma": 2.0,
  "focal_alpha": 0.25,
  "focal_tversky_weight": 0.25,
  "ft_alpha": 0.7,
  "ft_beta": 0.3,
  "ft_gamma": 0.75,
  "l1_weight": 0.1,
  "boundary_weight": 0.2
}

