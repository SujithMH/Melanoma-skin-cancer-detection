# 🔬 Melanoma-HPO: Hybrid Deep Learning with Meta-Heuristic Hyperparameter Optimisation

> **Multi-task skin lesion classification** using a fused EfficientNet-B1 + DeiT-Small architecture,
> optimised via Grey Wolf Optimiser (GWO), Genetic Algorithm (GA), and Random Search,
> with an explainable AI (XAI) Streamlit app powered by Grad-CAM++.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Pipeline](#project-pipeline)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Hyperparameter Search Space](#hyperparameter-search-space)
- [HPO Algorithms](#hpo-algorithms)
- [Results](#results)
- [Explainability (XAI)](#explainability-xai)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Running the App](#running-the-app)
- [Known Limitations & Design Decisions](#known-limitations--design-decisions)

---

## Overview

This project investigates whether **meta-heuristic hyperparameter optimisation** can significantly
improve a hybrid CNN–Transformer network for dermoscopic skin lesion classification.

The core contributions are:

1. **HybridDeepNet** — a multi-task model that jointly predicts:
   - **7-class lesion type** (akiec, bcc, bkl, df, mel, nv, vasc)
   - **4-tier severity grade** (Benign → Pre-malignant → NMSC → Melanoma)

2. **HPO Harness** — a proxy-based evaluation harness that makes running 60+ full HPO trials
   computationally feasible by training on a 35% data fraction for 5 epochs.

3. **Three search algorithms** compared head-to-head: Random Search, GWO, and GA.

4. **Focal Loss re-training** — the winning configuration is retrained with class-balanced Focal
   Loss to address the severe HAM10000 class imbalance (nv = 67% of data).

5. **XAI App** — a Streamlit app with Grad-CAM++ explanations, prior-corrected probabilities,
   and asymmetric malignancy thresholds designed for clinical screening use.

---

## Project Pipeline

The project is structured as 7 sequential phases:

```
Phase A  →  Phase B  →  Phase C  →  Phase D  →  Phase E  →  Phase F  →  Phase G
DataLoader  Baselines  Hybrid MTL  Noise Study   HPO Run   Final Train  GradCAM XAI
  Test       (CNN+ViT)  (MTL Arch)  (Variance)  (GWO/GA/RS) (Focal Loss) Visualise
```

| Phase | Script | Purpose |
|-------|--------|---------|
| A | `run_phase_a.py` | DataLoader speed benchmark & sanity check |
| B | `run_phase_b.py` | Train EfficientNet-B1 and DeiT-Small baselines (15 epochs) |
| C | `run_phase_c.py` | Train HybridDeepNet with concat fusion (15 epochs) |
| D | `run_phase_d.py` | Noise floor study — measure proxy objective variance |
| E | `run_phase_e.py` | Run HPO: `--algo [rs\|gwo\|ga] --seed [42\|101\|2024]` |
| F | `run_phase_f.py` | Retrain best config (CrossEntropy) for 20 epochs |
| F-Focal | `run_phase_f_focal.py` | Retrain best config with **Focal Loss** for 20 epochs |
| G | `run_phase_g.py` | Generate Grad-CAM heatmap visualisations on val set |

---

## Architecture

### HybridDeepNet (`src/models/hybrid.py`)

```
Input (224×224×3)
    │
    ├─── EfficientNet-B1 (timm) ──► f_cnn [B, 1280]
    │      pretrained backbone
    │
    └─── DeiT-Small (timm) ──────► f_vit [B, 384]
           pretrained backbone
               │
         CONCAT fusion: [B, 1664]
               │
         Shared FC Head:
           Linear(1664 → fc_hidden)
           BatchNorm1d
           GELU
           Dropout(p)
               │
         ┌─────┴──────┐
         │            │
    cls_head       sev_head
  Linear(→7)    Linear(→4)
         │            │
   7-class          4-tier
   softmax        severity
```

**Key design choices:**
- **Concat fusion** (chosen by HPO over additive fusion) preserves both feature spaces independently
- **Shared FC head** forces the two tasks to share a common representation
- **Differential learning rates** — backbone LR = `head_lr × backbone_lr_mult` (≈14× smaller) to
  avoid catastrophic forgetting of ImageNet weights

### Branches (`src/models/branches.py`)
Standalone EfficientNet-B1 and DeiT-Small wrappers used as baselines in Phase B.

### Loss Functions (`src/models/losses.py`)

**`HybridLoss`** (Phase C/F):
```
L_total = L_cls (CrossEntropy) + sev_weight × L_sev (CrossEntropy, ignore_index=-1)
```

**`HybridLossFocal`** (Phase F-Focal):
```
L_cls  = Focal Loss (α per class, γ=2.0) — addresses 67% nv dominance
L_total = L_cls + sev_weight × L_sev
```
`vasc` is masked in severity loss (`ignore_index=-1`) because it has no meaningful severity tier.

---

## Dataset

**HAM10000** — Human Against Machine with 10,000 training images from ISIC Archive.

### Class Distribution (train split)

| Class | Label | Severity | Train Samples | % |
|-------|-------|----------|--------------|---|
| Melanocytic Nevi | `nv` | 0 — Benign | 4,718 | 67.4% |
| Melanoma | `mel` | 3 — Malignant | 770 | 11.0% |
| Benign Keratosis | `bkl` | 1 — Pre-malignant | 741 | 10.6% |
| Basal Cell Carcinoma | `bcc` | 2 — NMSC | 381 | 5.4% |
| Actinic Keratosis | `akiec` | 2 — NMSC | 207 | 3.0% |
| Vascular Lesion | `vasc` | masked | 96 | 1.4% |
| Dermatofibroma | `df` | 1 — Benign | 89 | 1.3% |

> ⚠️ The severe imbalance (`nv` = 67%) is the dominant challenge. Even with Focal Loss, the model
> develops a strong prior toward `nv`. The inference app corrects for this explicitly.

### Data Splits
| Split | File | Samples |
|-------|------|---------|
| Train | `splits/train.csv` | 7,002 |
| Val | `splits/val.csv` | ~1,500 |
| Test | `splits/test.csv` | ~1,500 |

### Preprocessing (`src/data/dataset.py`)

**Training:**
```
RandomResizedCrop(224, scale=0.8–1.0)
HorizontalFlip(p=0.5) + VerticalFlip(p=0.5)
Affine(scale, translate, rotate)(p=0.5)
ColorJitter(p=0.5)
CoarseDropout(p=0.3)
Normalize(ImageNet mean/std)
```

**Validation/Inference:**
```
Resize(256, 256)
CenterCrop(224, 224)
Normalize(ImageNet mean/std)
```

---

## Hyperparameter Search Space

The HPO searches a **7-dimensional continuous space** `[0,1]^7` decoded to real hyperparameters:

| Dim | Hyperparameter | Range | Scale |
|-----|---------------|-------|-------|
| 0 | `head_lr` | 1e-5 – 1e-3 | Log |
| 1 | `backbone_lr_mult` | 0.05 – 1.0 | Linear |
| 2 | `weight_decay` | 1e-5 – 1e-2 | Log |
| 3 | `dropout` | 0.1 – 0.6 | Linear |
| 4 | `fc_hidden` | 128 – 1024 (step 64) | Linear+Round |
| 5 | `fusion_type` | concat / add | Categorical |
| 6 | `sev_weight` | 0.1 – 1.0 | Linear |

### Proxy Objective (`src/hpo/objective.py`)
Each candidate is evaluated on **35% of the training data for 5 epochs** — a proxy that
correlates with full training while being ~7× faster. The proxy objective is
**validation Macro-F1** (averaged across 3 seeds in the noise study).

---

## HPO Algorithms

### Grey Wolf Optimiser (GWO) — `src/hpo/gwo.py`
Mimics the leadership hierarchy of grey wolves (α, β, δ, ω).
- **Population:** 6 wolves
- **Iterations:** 10
- **Total evaluations:** 60
- The `a` parameter decreases linearly from 2 → 0, balancing exploration vs exploitation

```python
# Position update rule (per dimension)
X_new = (X_alpha + X_beta + X_delta) / 3  # Average of three leader influences
```

### Genetic Algorithm (GA) — `src/hpo/ga.py`
- **Population:** 6 individuals
- **Generations:** 10
- Tournament selection, single-point crossover, Gaussian mutation

### Random Search (RS) — `src/hpo/random_search.py`
Uniform random sampling in `[0,1]^7` — used as the baseline comparator.

### HPO Runner (`src/hpo/runner.py`)
- Logs all evaluations to JSONL files in `logs/`
- Caches results by vector hash (crash-safe resume)
- Supports multi-seed runs

---

## Results

### Phase B — Baseline Models (15 epochs)

| Model | Val Macro-F1 |
|-------|-------------|
| EfficientNet-B1 (standalone) | ~0.698 |
| DeiT-Small (standalone) | ~0.65 |

### Phase C — Hybrid MTL (15 epochs, default config)

| Model | Val Macro-F1 |
|-------|-------------|
| HybridDeepNet (concat, default LR) | **0.7030** |

### Phase F — HPO-Optimised Training

| Config | Val Macro-F1 |
|--------|-------------|
| CrossEntropy, 20 epochs | 0.7163 |
| **Focal Loss, 20 epochs** | **~0.72** (best checkpoint) |

### Winning Hyperparameters (from GWO)

```python
{
    'head_lr':          9.8e-4,
    'backbone_lr_mult': 0.14,      # Backbone LR = head_lr × 0.14
    'weight_decay':     0.01,
    'dropout':          0.357,
    'fc_hidden':        448,
    'fusion_type':      'concat',
    'sev_weight':       0.428
}
```

---

## Explainability (XAI)

The Streamlit app (`app.py`) implements a full clinical XAI pipeline:

### 1. Prior-Corrected Inference
The model's softmax output is biased by the training distribution (nv = 67%).
We correct for this using Bayes' theorem:

```python
corrected = softmax_probs / training_priors
corrected = corrected / corrected.sum()
```

This is equivalent to assuming a uniform prior at inference time — appropriate for a screening
tool where any lesion type may appear.

### 2. Asymmetric Malignancy Thresholds
Standard argmax misses melanoma cases where `mel = 28%` but `nv = 40%`.
We instead independently check each high-risk class against a low threshold:

| Class | Threshold (corrected prob) | Rationale |
|-------|--------------------------|-----------|
| `mel` | **20%** | Most dangerous — lowest threshold |
| `bcc` | **25%** | Basal cell carcinoma |
| `akiec` | **30%** | Pre-malignant |

Any class exceeding its threshold triggers a malignancy flag, regardless of whether it is argmax.

### 3. Grad-CAM++ XAI Panel
4-panel matplotlib visualisation:

| Panel | Content |
|-------|---------|
| ① Original | Image with attention centroid (red ✕) |
| ② GradCAM++ → flagged class | What drives the malignancy suspicion |
| ③ GradCAM++ → `nv` baseline | What drives a benign prediction |
| ④ Binary attention mask | Top 25% activations highlighted |

**Technical choices:**
- **GradCAM++** (not plain GradCAM) — better localisation for multi-instance cases
- **Target layer:** `blocks[5][-1].conv_pwl` — 14×14 spatial maps (2× finer than `conv_head`)
- **`aug_smooth=True` + `eigen_smooth=True`** — averages augmented views + PCA denoising
- **Explicit class targeting** — always explains the clinically relevant class, not raw argmax
- **Watermark warning** — auto-detects if attention centroid is near image borders (where
  VisualDx watermarks appear in HAM10000 images)

---

## Project Structure

```
melanoma-hpo/
│
├── app.py                          # Streamlit XAI app (main entry point)
│
├── src/
│   ├── data/
│   │   ├── dataset.py              # MelanomaDataset, augmentation pipelines
│   │   ├── preprocess.py           # Image caching / resizing utilities
│   │   └── splits.py              # Train/val/test CSV generation
│   │
│   ├── models/
│   │   ├── hybrid.py               # HybridDeepNet (EfficientNet + DeiT, MTL)
│   │   ├── branches.py             # Standalone CNN & ViT baselines
│   │   └── losses.py               # HybridLoss, HybridLossFocal, FocalLoss
│   │
│   ├── hpo/
│   │   ├── encoding.py             # 7D vector → hyperparameter dict decoder
│   │   ├── objective.py            # Proxy training harness (35% data, 5 epochs)
│   │   ├── runner.py               # Evaluation logger + crash-safe resume
│   │   ├── gwo.py                  # Grey Wolf Optimiser
│   │   ├── ga.py                   # Genetic Algorithm
│   │   └── random_search.py        # Random Search baseline
│   │
│   └── train.py                    # Generic training loop (Phase B baselines)
│
├── run_phase_a.py                  # DataLoader benchmark
├── run_phase_b.py                  # Baseline training (CNN + ViT)
├── run_phase_c.py                  # Hybrid MTL training
├── run_phase_d.py                  # Noise floor study
├── run_phase_e.py                  # HPO execution (rs/gwo/ga + seed)
├── run_phase_f.py                  # Final training (CrossEntropy)
├── run_phase_f_focal.py            # Final training (Focal Loss) ← used in app
├── run_phase_g.py                  # Grad-CAM visualisation on val set
├── generate_final_report.py        # Aggregates HPO logs into report
│
├── splits/
│   ├── train.csv                   # 7,002 training samples
│   ├── val.csv                     # ~1,500 validation samples
│   └── test.csv                    # ~1,500 test samples
│
├── logs/                           # HPO JSONL logs (per algo × seed)
├── train_results/                  # Training output logs
├── data/cache/                     # Preprocessed image cache (not in repo)
│
├── best_optimized_hybrid.pth       # Best weights (CrossEntropy training)
├── best_optimized_hybrid_focal.pth # Best weights (Focal Loss) ← used by app
│
├── confusion_matrix_01.png         # Validation confusion matrix
├── gradcam_results.png             # Phase G Grad-CAM sample output
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.12
- CUDA-compatible GPU recommended (tested with 4GB VRAM, NVIDIA)
- CUDA 11.8+ / cuDNN

### 1. Clone & create virtual environment
```bash
git clone https://github.com/<your-username>/melanoma-hpo.git
cd melanoma-hpo
python -m venv mel-venv-3.12
# Windows
mel-venv-3.12\Scripts\activate
# Linux/Mac
source mel-venv-3.12/bin/activate
```

### 2. Install PyTorch (CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install remaining dependencies
```bash
pip install timm streamlit albumentations scikit-learn opencv-python pandas numpy tqdm
pip install grad-cam  # pytorch-grad-cam
```

### 4. Prepare the dataset
Download [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
from Kaggle and place images in `data/cache/` as `<image_id>.jpg`.

The `splits/` CSV files are already included — they contain `image_id` and `dx` columns.

> **Note:** `data/cache/` is not committed to git (listed in `.gitignore`). The model weights
> (`.pth` files) are also large (~110MB each) — consider using Git LFS or hosting them separately.

---

## Running the Pipeline

### Phase A — Test DataLoader
```bash
python run_phase_a.py
```

### Phase B — Train Baselines
```bash
python run_phase_b.py
```

### Phase C — Train Hybrid MTL
```bash
python run_phase_c.py
```

### Phase D — Noise Floor Study
```bash
python run_phase_d.py
```

### Phase E — HPO Runs
Run all 9 combinations (3 algorithms × 3 seeds):
```bash
python run_phase_e.py --algo gwo --seed 42
python run_phase_e.py --algo gwo --seed 101
python run_phase_e.py --algo gwo --seed 2024
python run_phase_e.py --algo ga  --seed 42
python run_phase_e.py --algo ga  --seed 101
python run_phase_e.py --algo ga  --seed 2024
python run_phase_e.py --algo rs  --seed 42
python run_phase_e.py --algo rs  --seed 101
python run_phase_e.py --algo rs  --seed 2024
```
> Each run takes ~7 hours (60 evaluations × ~7 min/eval on a mid-range GPU).

### Phase F — Final Training
```bash
# Standard CrossEntropy
python run_phase_f.py

# Focal Loss (recommended — used by the app)
python run_phase_f_focal.py
```

### Phase G — Grad-CAM Visualisation
```bash
python run_phase_g.py
```

---

## Running the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**App features:**
- Upload any dermoscopic image (JPG/PNG)
- See the **binary verdict** (MALIGNANT / BENIGN) with severity tier
- View **prior-corrected vs raw model probabilities** side by side
- Toggle **Grad-CAM++ XAI panel** (4-panel: original, flagged class heat, nv baseline, binary mask)
- Attention quality warnings if the model focuses on image borders/watermarks

---

## Known Limitations & Design Decisions

### Class Imbalance
HAM10000 is heavily `nv`-dominated (67%). Even with Focal Loss, the raw model output is biased.
The app applies Bayesian prior correction at inference time rather than retraining, as a pragmatic
fix. Proper solutions would include oversampling/undersampling or a balanced dataset.

### Watermark Leakage
Some HAM10000 images (particularly from VisualDx) contain visible watermarks. Grad-CAM++
occasionally highlights these watermarks as discriminative features — a form of spurious
correlation that would require dataset cleaning to resolve permanently.

### Proxy Fidelity
The HPO proxy (35% data, 5 epochs) is a noisy estimate of full performance. The noise study
(Phase D) confirmed variance ≤ ±0.02, which was deemed acceptable. However, optimal
hyperparameters from proxy training may not perfectly transfer to full training.

### CPU Inference
Running `app.py` on CPU is supported but slow (model is ~110MB, inference takes ~10–30s per image
without CUDA). `torch.amp.autocast` is only applied when CUDA is available.

---

## Citation / Acknowledgements

- **Dataset:** [HAM10000](https://doi.org/10.1038/sdata.2018.161) — Tschandl et al., 2018
- **EfficientNet-B1:** Tan & Le, EfficientNet (ICML 2019)
- **DeiT-Small:** Touvron et al., Training data-efficient image transformers (ICML 2021)
- **GWO:** Mirjalili et al., Grey Wolf Optimizer (Advances in Engineering Software, 2014)
- **Grad-CAM++:** Chattopadhyay et al., Grad-CAM++ (WACV 2018)
- **pytorch-grad-cam:** [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- **timm:** [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

---

> ⚕️ **Medical Disclaimer:** This tool is a research prototype and is **not** approved for clinical
> use. All outputs should be reviewed by a qualified dermatologist. Do not use for diagnosis.
