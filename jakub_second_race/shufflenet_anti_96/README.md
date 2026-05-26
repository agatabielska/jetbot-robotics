# shufflenet_anti_96 ⭐

**Mirror działającego shufflenet drużynowego** (`first_race_errors/resized-no-augmented-jetbot`)
+ nasz anti-bias stack.

## Architektura
- **Backbone**: `ShuffleNetV2 x0.5` pretrained ImageNet (torchvision)
- **Head**: `Linear(1024→64) → ReLU → Linear(64→2) → Tanh`
- **Input**: 96×96 RGB, ImageNet-normalized
- **Output**: 2 wartości `[forward, left]` w strict `(-1, 1)`
- **Params**: 407,522 (pod budżetem PUT 600k)
- **ONNX**: opset 11 ✓

## Czego ich shufflenet NIE miał (a my mamy)
1. **HorizontalFlip + target_swap** — adresuje brak `left=+1.0` w dataset
2. **WeightedRandomSampler** — 5× oversampling rzadkich skrętów (7 binów `|left|`)
3. **label_shift=+2** (~200ms predict-ahead) — adresuje przedwczesny skręt
4. **Strict clip ±0.999** w postprocess
5. **EMA smoothing runtime** — adresuje zygzak na prostych

## Trening
- **val_loss**: 0.235 (best epoch 18)
- **Dataset**: 6064 train / 1470 val (Phase 1 hold-out split + blacklist)
- **Loss**: Huber `delta=0.1`
- **Optimizer**: AdamW `lr=5e-4, wd=1e-5`
- **Scheduler**: CosineAnnealingLR
- **Epochs**: max 40, patience=10 (early stop @ epoka 28)

Reproducja: `python3 train.py` (wymaga venv z Phase 2 + GPU CUDA).

## Inference
- **preprocess**: BGR uint8 → RGB → /255 → ImageNet norm → resize 96×96 → (1,3,96,96)
- **postprocess**: clip strict ±0.999
- **EMA**: `inference.ema_alpha=0.3` (tunable w config.yml)

## Kiedy używać
- **Domyślny wybór** — najszybszy inference (~5ms na Jetson Nano), najbliższy z designem do działającego shufflenet drużynowego
- Adresuje OBA znane defekty shufflenet drużynowego (zygzak + przedwczesny skręt)

## Common failure modes
- Jeśli **silne zygzakowanie** → zmniejsz `inference.ema_alpha` do 0.2 lub 0.1
- Jeśli **reaguje za wolno** → zwiększ `inference.ema_alpha` do 0.5 lub 0.7
- Jeśli **dryfuje w bok** → uruchom `../calibration/calibrate.py`
