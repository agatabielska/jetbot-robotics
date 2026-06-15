# Jakub Górniak Team

## Agata Bielska, Jakub Radziejewski, Jakub Jęsiek, Kacper Kuźnik, Aleksander Hański

**During classes we created many models, they were almost all based on ResNet or PilotNet architectures.**


## Other Approaches

### 1. Transfer Learning - ResNet18 Pretrained

**Path**: `experiments/train_pretrained_cnn.py`

Standard ImageNet pretrained ResNet18 with a custom regression head (Linear → Tanh → 2 outputs).

| Setting | Value |
|---|---|
| LR | 1e-4 (Adam) |
| Augmentation | RandomCrop, ColorJitter, GaussianBlur, RandomErasing |
| Anti-bias | WeightedRandomSampler weighted by label magnitude |
| Loss | Val2WeightedMSELoss (steering gradient up-weighted) |

---

### 2. Dual-Head ResNet18 - Classification + Regression

**Path**: `experiments/dualhead.py`

ResNet18 backbone with two separate heads:
- **Head 1**: regression for `forward` (Linear → Tanh)
- **Head 2**: 3-class classifier for `left` direction ({left, straight, right})

| Setting | Value |
|---|---|
| LR | 1e-4 (AdamW), ReduceLROnPlateau |
| Dropout | 0.5 |
| Weight decay | 1e-2 |
| Augmentation | RandomRotation(15°), RandomPerspective, ColorJitter, GaussianBlur, RandomErasing |
| Anti-bias | Inverse-frequency class weighting for turn classifier |
| Loss | MSE(forward) + CrossEntropy(left direction) |

---

### 3. Dual Classifier - ResNet18, Speed + Turn Both as Classes

**Path**: `experiments/dualclassifer.py`

Both outputs treated as classification problems:
- **Speed head**: 3 classes — slow (0.0), medium (0.5), fast (1.0)
- **Turn head**: 3 classes — left (−1), straight (0), right (+1)
- Shared Linear(512→256) + BatchNorm + ReLU + Dropout(0.3) layer

| Setting | Value |
|---|---|
| LR | 1e-3 (AdamW), ReduceLROnPlateau |
| Dropout | 0.3 |
| Augmentation | RandomRotation(5°), RandomPerspective, ColorJitter, GaussianBlur, RandomErasing, optional CLAHE |
| Anti-bias | Inverse-frequency class weighting for both heads |
| Loss | CrossEntropy × 2 |

---

### 4. Custom PilotNet

**Path**: `experiments/Kacper-Solution/`

Custom PilotNet-inspired architecture:
- 5 conv layers (32→32→64→128→128) with BatchNorm
- Fully connected regression head

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Augmentation | HSV-space color, Gaussian noise, shadow cutout, horizontal flip (negate steering) |
| Anti-bias | Horizontal flip oversampling |


---

### 5. PilotNet - Original NVIDIA Architecture

**Path**: `experiments/pilotnet/`

Faithful NVIDIA PilotNet implementation:
- 5 conv layers (24→36→48→64→64), ELU activations
- 4 FC layers (100→50→10→2), Tanh output

| Setting | Value |
|---|---|
| LR | 1e-3 (AdamW) + 3-epoch linear warmup + cosine decay |
| Anti-bias | WeightedRandomSampler on 10 steering bins |
| Loss | steer_MSE × 1.0 + throttle_MSE × 0.3 |

---

### 6. 6 Variants of PilotNet-style backbone

**Path**: `experiments/jakub_second_race/`

All variants share: PilotNet-style backbone, AdamW, cosine LR, ONNX opset 11 export, runtime EMA smoothing (α=0.3).

| Model | Architecture | Input | Params | Key Difference | Val Loss |
|---|---|---|---|---|---|
| `pilotnet_anti` | PilotNet 224 | 224×224 | ~195k | label_shift=2 (200ms lookahead) | 0.213 |
| `pilotnet_fl_shift1` | PilotNet 224 | 224×224 | ~195k | label_shift=1 (100ms lookahead) | 0.216 |
| `mobilenetv3_fl_pretrained` | MobileNetV3-Small (pretrained) | 224×224 | 1.08M | ImageNet backbone, lr=5e-4 | **0.170** |
| `shufflenet_anti_96` | ShuffleNetV2 x0.5 (pretrained) | 96×96 | ~407k | Smallest input, fastest inference | 0.235 |
| `shufflenet_anti_224` | ShuffleNetV2 x0.5 (pretrained) | 224×224 | ~407k | Same as above, larger input | 0.249 |
| `dualhead_class_reg` | PilotNet + dual head | 224×224 | ~240k | Left classified (7 bins) + forward regressed | 0.211 |

All use WeightedRandomSampler on 7 `|left|` bins (except `dualhead_class_reg` which uses CrossEntropy + smooth L1).

---

### 7. Other models

**Path**: `experiments/jetbot_racing_bundle/`

| Model | Key Idea | Anti-bias Strategy |
|---|---|---|
| `pilotnet_fl_wsampler` | Baseline PilotNet 224 | WeightedRandomSampler, 7 bins |
| `pilotnet_fl_wloss` | Same arch, weighted loss instead of sampler | Per-dim Huber loss [forward×1.0, left×3.0] |
| `pilotnet_fl_shift1` | Temporal lookahead to fix late reactions | WS 7 bins + label_shift=1 |
| `pilotnet_motor_wsampler` | Predict (motor_L, motor_R) directly | WS on motor differential |
| `dualhead_class_reg` | Classify left direction, regress forward | CrossEntropy(7-bin left) + smooth L1 |
| `mobilenetv3_fl_pretrained` | Pretrained backbone for richer features | WS 7 bins, ImageNet norm |

All: lr=1e-3, AdamW, cosine schedule, batch_size=64, epochs=50, early stopping patience=10.

### 8. Pure computer vision approach

Attempt to implement only binary thresholding and morphological operations in order to find the 
center of the road. It was not great enough to be included in experiments in final repo.

---


## Anti-Bias Techniques Tried

| Technique | Where Used | Notes |
|---|---|---|
| WeightedRandomSampler (steering bins) | Most PilotNet variants | 7–10 bins on `|left|`, ~5× oversampling of rare turns |
| Per-dim weighted loss | `pilotnet_fl_wloss` | left gradient 3× stronger than forward |
| Dual-head classification | `dualhead_class_reg` variants | Turns treated as discrete decisions, not regression |
| Inverse-frequency class weights | ResNet dual-head/dual-classifier | Applied to CrossEntropy loss |
| Label shift (temporal lookahead) | `pilotnet_anti`, `pilotnet_fl_shift1` | Addresses late reactions in curves |
| Horizontal flip + steering negation | Most experiments | Balances left/right turn distribution |
| HSV augmentation + shadow cutout | Kacper-Solution | Improves lighting robustness |