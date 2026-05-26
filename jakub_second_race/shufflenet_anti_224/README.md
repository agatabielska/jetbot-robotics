# shufflenet_anti_224

**Wariant shufflenet_anti z większym inputem 224×224** — bogatsze cechy, ale wolniejszy inference.

## Architektura
- **Backbone**: `ShuffleNetV2 x0.5` pretrained ImageNet (torchvision)
- **Head**: `Linear(1024→64) → ReLU → Linear(64→2) → Tanh`
- **Input**: 224×224 RGB, ImageNet-normalized (jak natywny ImageNet input)
- **Output**: 2 wartości `[forward, left]` w strict `(-1, 1)`
- **Params**: 407,522 (identyczne z 96 — CNN params nie zależą od input size)
- **ONNX**: opset 11 ✓

## Trening
- **val_loss**: 0.249 (best epoch 8)
- Identyczne hyperparams jak `shufflenet_anti_96`, tylko `input_size: 224`
- 94s na GPU CUDA

## Kiedy używać
- Backup dla `shufflenet_anti_96` jeśli ten ma problemy z generalizacją
- 224 jest natywnym rozmiarem ImageNet — pretrained features mogą być lepiej dopasowane
- Trade-off: ~5.76× więcej pikseli → wolniejszy inference na Jetson Nano

## Inference
- **preprocess**: BGR → RGB → /255 → ImageNet norm → resize 224×224 → (1,3,224,224)
- **postprocess**: clip ±0.999
- **EMA**: `inference.ema_alpha=0.3`

## Uwaga
W naszych benchmark val_loss 224 (0.249) > 96 (0.235) — może wynikać z wcześniejszego early stopping (best_epoch=8) lub większej trudności fit na bogatszym inpucie z małym datasetem (6k samples). Test empiryczny na torze pokaże.
