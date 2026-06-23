# ID Card Presentation Attack Detection (PAD)

A PyTorch framework for detecting **replay spoofing attacks** on ID card images — specifically screens (phone/tablet/monitor) used to present a previously captured ID card to a camera.

The model is a **ConvNeXt backbone** augmented with a **MinVision-inspired auxiliary frequency branch** that regresses an FFT magnitude map from an intermediate feature map, forcing the network to learn frequency-domain artifacts characteristic of screen replay attacks (moiré patterns, pixel-grid regularity).

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experiment Outputs](#experiment-outputs)
- [Metrics](#metrics)
- [Extending the Project](#extending-the-project)

---

## How It Works

### The Spoof Problem
Screen replay attacks present a real ID card image on a digital display in front of a camera. The challenge is that the ID content itself looks legitimate — the discriminating signal lies in:
- **Moiré patterns** from screen pixel grids interfering with the camera sensor
- **Unnatural color profiles** and saturation from display backlights
- **Frequency domain regularity** — screen pixels introduce periodic structures invisible to the eye but clear in the FFT spectrum

### The Model
The architecture has two outputs trained jointly:

```
Input Image (3×H×W)
       │
  ConvNeXt Backbone
       │
  ┌────┴──────────────────────┐
  │ (hook at stage N)         │
  │                           ▼
  │              Auxiliary Frequency Branch
  │                           │
  │               freq_pred (1×fH×fW)  ──► MSE Loss ◄── FFT map (ground truth)
  │
  ▼
Classification Head
       │
  cls_logit (scalar) ──────────────────► BCE Loss ◄── label (0=real, 1=replay)
       │
  Total Loss = λ_cls · L_BCE + λ_freq · L_MSE
```

The auxiliary branch is **only active during training**. At inference, only the classification head is used.

### Frequency Map (Ground Truth)
For each training image, the ground-truth frequency supervision target is computed as:

```
RGB → Y (luminance) → 2D FFT → shift zero-freq to center
     → log(1 + |F|) → per-image min-max normalize → resize to (fH, fW)
```

This log-scaled FFT magnitude map is precomputed in the dataset class before any augmentation is applied, ensuring the frequency map reflects the original capture artifact.

---

## Project Structure

```
project/
│
├── config.py           # All hyperparameters and paths (single source of truth)
├── dataset.py          # IDCardDataset class + transform pipelines + dataloader factory
├── model.py            # ConvNeXtWithFreqBranch + AuxFrequencyBranch + build_model()
├── train.py            # Training loop (fresh start or resume)
├── test.py             # Evaluation script with ROC curve plotting
│
├── utils/
│   ├── __init__.py
│   ├── frequency.py    # FFT magnitude map computation
│   ├── metrics.py      # MetricsAccumulator — EER, HTER, AUC, FAR, FRR
│   └── logger.py       # ExperimentLogger — stdout + .log file + metrics.csv
│
├── dataset/
│   ├── train/
│   │   ├── real/
│   │   └── replay/
│   ├── val/
│   │   ├── real/
│   │   └── replay/
│   └── test/
│       ├── real/
│       └── replay/
│
└── experiments/
    └── 2026-06-23_14-32-05/        # one directory per run, named by start time
        ├── config_snapshot.yaml    # exact config that produced this run
        ├── train.log
        ├── test.log
        ├── metrics.csv
        ├── checkpoints/
        │   ├── best.pth            # lowest val HTER
        │   └── last.pth            # most recent epoch
        └── plots/
            └── roc_curve.png
```

---

## Installation

**Python 3.9+** is required.

```bash
# Clone the repository
git clone https://github.com/your-org/id-card-pad.git
cd id-card-pad

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pyyaml>=6.0
```

> **GPU note:** The `torch` line above installs the CPU build. For CUDA support visit [pytorch.org](https://pytorch.org/get-started/locally/) and install the wheel matching your CUDA version.

---

## Dataset Preparation

Organize your images into the following layout **before** running any script:

```
dataset/
├── train/
│   ├── real/       ← genuine ID card captures
│   └── replay/     ← screen replay attack captures
├── val/
│   ├── real/
│   └── replay/
└── test/
    ├── real/
    └── replay/
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

**Recommended split:** 70% train / 15% val / 15% test, stratified by capture device to prevent device-identity leakage into the metrics.

**Class imbalance** is handled automatically by `WeightedRandomSampler` (enabled by default in `config.py`). If your classes are balanced, you can disable this with `use_weighted_sampler: false`.

---

## Configuration

All settings are controlled from `config.py`. You can either edit the defaults directly or supply a YAML override file at runtime.

### Key parameters

| Section | Parameter | Default | Description |
|---|---|---|---|
| `model` | `backbone` | `convnext_tiny` | ConvNeXt variant (`tiny` / `small` / `base`) |
| `model` | `aux_branch_stage` | `2` | Stage (1–4) to tap for the frequency branch |
| `model` | `pretrained` | `True` | Load ImageNet weights for the backbone |
| `frequency` | `freq_map_size` | `(28, 28)` | Spatial size of FFT supervision map |
| `data` | `image_size` | `(224, 224)` | Input resolution fed to the model |
| `data` | `batch_size` | `32` | Training batch size |
| `training` | `num_epochs` | `50` | Maximum training epochs |
| `training` | `learning_rate` | `1e-4` | Initial learning rate (AdamW) |
| `training` | `lambda_cls` | `1.0` | Weight for classification BCE loss |
| `training` | `lambda_freq` | `0.5` | Weight for frequency MSE loss |
| `training` | `early_stopping_patience` | `10` | Epochs without val HTER improvement before stopping |
| `scheduler` | `scheduler_type` | `cosine` | LR schedule (`cosine` / `step` / `none`) |
| `scheduler` | `warmup_epochs` | `3` | Linear LR warmup epochs |

### Auxiliary branch stage guide

The `aux_branch_stage` setting controls where the frequency branch taps into the ConvNeXt backbone. For a 224×224 input image the spatial resolution at each stage is:

| Stage | Spatial size | Channels (tiny) | Character |
|---|---|---|---|
| 1 | 56 × 56 | 96 | Early, low-level texture |
| 2 | 28 × 28 | 192 | Mid-level — **recommended default** |
| 3 | 14 × 14 | 384 | Deeper, more semantic |
| 4 | 7 × 7 | 768 | Most abstract, least spatial |

> When you change `aux_branch_stage`, also update `freq_map_size` to match the corresponding spatial size in the table above.

### Using a YAML config file

```yaml
# my_experiment.yaml
model:
  backbone: convnext_small
  aux_branch_stage: 2

training:
  num_epochs: 80
  learning_rate: 5e-5
  lambda_freq: 0.3

data:
  batch_size: 16
```

```bash
python train.py --config my_experiment.yaml
```

---

## Training

### Start a new run

```bash
# With default config
python train.py

# With a custom config file
python train.py --config my_experiment.yaml
```

A new timestamped directory is created automatically under `experiments/`.

### Resume a previous run

```bash
python train.py --resume experiments/2026-06-23_14-32-05
```

The config snapshot from that experiment is reloaded automatically, and training continues from the last saved checkpoint. No manual config passing is required.

### Training output (live)

```
[2026-06-23 14:32:10]  Experiment directory : experiments/2026-06-23_14-32-05
[2026-06-23 14:32:10]  Device               : cuda
[2026-06-23 14:32:10]  Backbone             : convnext_tiny
...
[2026-06-23 14:33:01] [Epoch 01/50] [TRAIN]  loss=0.6821  eer=0.4102  far@eer=0.4098  frr@eer=0.4106  hter=0.4102  auc=0.6341
[2026-06-23 14:33:18] [Epoch 01/50] [VAL  ]  loss=0.5934  eer=0.3211  far@eer=0.3198  frr@eer=0.3224  hter=0.3211  auc=0.7512
[2026-06-23 14:33:18]  Checkpoint saved [BEST] → experiments/.../checkpoints/best.pth  (epoch 1)
```

---

## Evaluation

```bash
# Evaluate with best checkpoint (default)
python test.py --experiment experiments/2026-06-23_14-32-05

# Evaluate with a specific checkpoint
python test.py --experiment experiments/2026-06-23_14-32-05 --checkpoint last.pth
```

The script reloads the exact config snapshot from the experiment directory, ensuring the test pipeline is identical to training (same image size, normalization stats, etc.).

### Test output

```
── Test Results ──────────────────────────────────────────────
  AUC           : 0.9714
  EER           : 0.0681  (6.81%)
  FAR @ EER     : 0.0674
  FRR @ EER     : 0.0688
  HTER          : 0.0681  (6.81%)
  EER threshold : 0.4823
──────────────────────────────────────────────────────────────
ROC curve saved → experiments/2026-06-23_14-32-05/plots/roc_curve.png
```

---

## Experiment Outputs

Every experiment directory is fully self-contained:

| File | Description |
|---|---|
| `config_snapshot.yaml` | Exact config used — sufficient to reproduce the run |
| `train.log` | Timestamped log of every epoch and batch step |
| `test.log` | Timestamped test results |
| `metrics.csv` | All epoch metrics in CSV format — load with pandas for plotting |
| `checkpoints/best.pth` | Weights with the lowest validation HTER |
| `checkpoints/last.pth` | Weights from the final completed epoch |
| `plots/roc_curve.png` | ROC curve with EER point marked |

### Plotting training curves from CSV

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/2026-06-23_14-32-05/metrics.csv")
train = df[df["phase"] == "TRAIN"]
val   = df[df["phase"] == "VAL"]

plt.plot(train["epoch"], train["hter"], label="Train HTER")
plt.plot(val["epoch"],   val["hter"],   label="Val HTER")
plt.xlabel("Epoch"); plt.ylabel("HTER"); plt.legend(); plt.show()
```

---

## Metrics

All metrics are computed over the **full evaluation set**, never per-batch.

| Metric | Description |
|---|---|
| **AUC** | Area Under the ROC Curve. Higher is better. |
| **EER** | Equal Error Rate — threshold where FAR = FRR. Lower is better. Primary comparison metric in PAD literature. |
| **FAR @ EER** | False Acceptance Rate at the EER threshold — fraction of spoof samples accepted as real. |
| **FRR @ EER** | False Rejection Rate at the EER threshold — fraction of real samples rejected as spoof. |
| **HTER** | Half Total Error Rate = (FAR + FRR) / 2 at the EER threshold. Equivalent to EER when evaluated at the EER point. |
| **EER threshold** | The decision threshold (applied to sigmoid output) that achieves EER. Use this as the deployment threshold. |

The ROC curve and EER threshold are derived using scikit-learn's `roc_curve`. The EER is found as the point on the curve where `|FAR − FRR|` is minimized.

---

## Extending the Project

### Adding a new model

1. Define a new `nn.Module` class in `model.py`
2. Register it in the `build_model()` factory at the bottom of `model.py`
3. Set `model_name` in `config.py` (or your YAML) to the new name

### Changing the backbone

Edit `config.py`:
```python
backbone: str = "convnext_small"   # or "convnext_base"
```
Remember to update `freq_map_size` if you also change `aux_branch_stage`.

### Adding new augmentations

Edit `build_train_transforms()` in `dataset.py`. For JPEG compression simulation (highly recommended for this task), consider replacing `RandomAdjustSharpness` with `albumentations.ImageCompression`:

```bash
pip install albumentations
```

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(h, w),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.4),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])
```

---

## Reference

The auxiliary frequency branch design is inspired by:

> **Silent-Face-Anti-Spoofing** — MiniVision-AI  
> https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
