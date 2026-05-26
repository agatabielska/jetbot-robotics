# pilotnet_anti

**Combo wszystkich naszych anti-bias innowacji** w klasycznej architekturze PilotNet (NVIDIA end-to-end).
Najlepszy val_loss spośród naszych "anti" modeli (0.213).

## Architektura
- **Backbone**: PilotNet (5 Conv + AdaptiveAvgPool(3) + BatchNorm + ReLU) — `[../resized-no-augmented-jetbot/jetbot_models.py]` style
- **Head**: `Linear(576→100) → ReLU → Dropout(0.2) → Linear(100→50) → ReLU → Linear(50→10) → ReLU → Linear(10→2) → Tanh`
- **Input**: 224×224 BGR (no color conversion), `/255` normalization
- **Output**: 2 wartości `[forward, left]` w strict `(-1, 1)`
- **Params**: 195,102 (najmniejszy z naszych 6 modeli)
- **ONNX**: opset 11 ✓

## Anti-bias stack (wszystko razem)
1. **HorizontalFlip + target_swap** — adresuje brak `left=+1.0`
2. **WeightedRandomSampler** — 5× oversampling rzadkich skrętów
3. **label_shift=+2** (~200ms predict-ahead) — adresuje przedwczesny skręt
4. **Huber loss** `delta=0.1` — robust to saturation `forward=1.0` outliers
5. **Strict clip ±0.999** w postprocess
6. **EMA smoothing runtime** — adresuje zygzak

## Trening
- **val_loss**: **0.213** (best epoch 31) — najlepszy z naszych "anti" modeli
- **Dataset**: 6064 train / 1470 val
- **Loss**: Huber `delta=0.1`
- **Optimizer**: AdamW `lr=1e-3, wd=1e-5`
- **Scheduler**: CosineAnnealingLR
- **Epochs**: max 50, patience=10 (early stop @ epoka 41)
- **174s** na GPU CUDA

## Porównanie z istniejącymi PilotNet
| Model | val_loss | label_shift | params |
|---|---|---|---|
| pilotnet_fl_wsampler | 0.208 | 0 | 195k |
| pilotnet_fl_shift1 | 0.216 | 1 (~100ms) | 195k |
| **pilotnet_anti** | **0.213** | **2 (~200ms)** | **195k** |

`pilotnet_anti` ma agresywniejszy label_shift (200ms) niż `shift1` (100ms) → silniejsza predykcja przyszłości.

## Inference
- **preprocess**: BGR → /255 → resize 224×224 → (1,3,224,224) — bez ImageNet norm (custom train, nie pretrained)
- **postprocess**: clip ±0.999
- **EMA**: `inference.ema_alpha=0.3`

## Kiedy używać
- Jeśli `shufflenet_anti_96` ma za duży latency na Jetson Nano (pilotnet ~5ms vs shufflenet ~5-10ms)
- Jeśli pretrained features (ImageNet) źle generalizują na nasz tor LEGO city
- Jako sanity check — model trenowany od zera vs pretrained różne failure modes
