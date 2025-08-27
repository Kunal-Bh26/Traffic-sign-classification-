# Traffic Sign Classification — Custom CNN vs EfficientNet (TensorFlow/Keras)

**Goal.** Compare a hand‑crafted CNN with channel+spatial attention against a pretrained EfficientNetB0 for traffic sign recognition (43 classes, GTSRB‑style). We document **accuracy**, **data pipeline**, **training schedule**, and **trade‑offs**.

---

## Results (from the provided notebooks)

| Model            | Input Size | Batch | Epochs | Optimizer | Loss                           | Test Accuracy |
|------------------|-----------:|------:|-------:|-----------|--------------------------------|--------------:|
| Custom CNN       | 128, 128 | 32 | 50 | AdamW (lr=1e-3) | sparse_categorical_crossentropy | 98.80% |
| EfficientNetB0†  | 128, 128 | 32 | 15 | Adam (1e-3 → 1e-5 FT) | categorical_crossentropy       | 88.38% |

† EfficientNetB0 training is two‑phase: **frozen base** 15 epochs @ 1e‑3, then **fine‑tuning** 15 epochs @ 1e‑5.

*Numbers can vary with hardware, random seeds, and exact data split.*

---

## Repository Layout

```
notebooks/
├─ CustomCNN_model.ipynb
└─ EfficientNet_model.ipynb
README.md
requirements.txt
LICENSE
.gitignore
```

---

## Dataset

Both notebooks assume **43 traffic‑sign classes** (GTSRB). Update paths if you’re not running on Kaggle.

- **Custom CNN** uses a `DATA_PATH` like `/kaggle/input/gtsrb-german-traffic-sign` and builds a dataframe of image paths/labels by scanning class folders.
- **EfficientNetB0** constructs `train_df`, `val_df`, `test_df` and feeds them into `flow_from_dataframe` generators.

**Expected image size:** 128, 128 (resized before feeding the models).

> ⚠️ If running locally, set your own dataset root (e.g., `data/gtsrb/...`) in the notebooks’ path variables.

---

## Data Pipeline & Augmentations

### Custom CNN
- **Preprocessing:** CLAHE in LAB color space (Contrast Limited Adaptive Histogram Equalization) then normalized to `[0, 1]`.
- **Augmentations:** rotation (≈25°), zoom (0.2), shifts (0.15), shear (0.15), brightness jitter ([0.7, 1.3]), channel shift, `fill_mode='constant'`.
- **Labels:** integer class IDs → trained with `sparse_categorical_crossentropy`.

### EfficientNetB0
- **Preprocessing:** Keras `preprocess_input` from `tensorflow.keras.applications.efficientnet`.
- **Augmentations:** rotation (20°), zoom (0.2), width/height shift (0.15), shear (0.15), brightness range [0.8, 1.2], `fill_mode='nearest'`.
- **Labels:** one‑hot vectors → trained with `categorical_crossentropy`.

---

## Architectures

### Custom CNN (attention‑enhanced)
- **Building block:** `Conv2D → BatchNorm → Swish` (+ optional attention).
- **Attention:** **ChannelAttention** (GAP → bottleneck MLP with ratio=8 → sigmoid) and **SpatialAttention** (7×7 conv over concatenated channelwise avg/max).
- **Macro structure:**
  - Stem: 32 filters, stride 2 (no attention)
  - Block 1: 64 → MaxPool2D → Dropout 0.2
  - Block 2: 128 → 128 → MaxPool2D → Dropout 0.3
  - Block 3: 256 → 512 → MaxPool2D → Dropout 0.4
  - Head: GAP → Dense(512, swish, L2=1e‑4) → Dropout 0.5 → Dense(43, softmax)

- **Training:**
  - Optimizer: **AdamW** (lr=1e-3, weight_decay=1e‑4)
  - Loss: `sparse_categorical_crossentropy`
  - Epochs: **50**, batch size **32**
  - **Class weighting** for imbalance
  - Callbacks: EarlyStopping(patience=12) · ReduceLROnPlateau(patience=5) · ModelCheckpoint(`best_model.h5`)

### EfficientNetB0 (transfer learning)
- **Backbone:** `EfficientNetB0(include_top=False, weights='imagenet')`
- **Head:** GAP → Dense(256, ReLU) → Dropout(0.3) → Dense(43, softmax)
- **Training schedule:**
  1) **Feature extraction:** base frozen, 15 epochs, Adam(lr=1e‑3)
  2) **Fine‑tuning:** unfreeze base, 15 epochs, Adam(lr=1e‑5)
  - Callbacks: EarlyStopping(patience=6) · ReduceLROnPlateau(patience=3) · ModelCheckpoint(`best_model.h5`)

---

## How to Reproduce

> Ensure the dataset path variables match your environment (Kaggle vs local).

1. **Create a virtual environment and install deps**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the notebooks**
   - `notebooks/CustomCNN_model.ipynb` → trains for 50 epochs; saves `best_model.h5`; prints **Base Test Accuracy**.
   - `notebooks/EfficientNet_model.ipynb` → 15 + 15 epochs (fine‑tuning); saves `best_model.h5`; prints **Test Accuracy**.

3. **(Optional) Save artifacts**
   - Put checkpoints under `results/checkpoints/`
   - Plots under `results/figures/`
   - Metrics JSON/CSV under `results/metrics/`

---

## Environment

- **Framework:** TensorFlow/Keras
- **Core packages:** see `requirements.txt`
- Suggested: set random seeds for reproducibility; record `pip freeze > results/env.txt` and note your hardware (GPU/CPU).

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Acknowledgments

- GTSRB dataset
- Keras Applications: EfficientNetB0
- TensorFlow/Keras community
