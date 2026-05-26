# mobilenetv3_fl_pretrained

**Wariant**: **torchvision MobileNetV3-Small z ImageNet pretrained weights**, head wymieniona na 2-output tanh regressor. WeightedRandomSampler. **rgb_imagenet preprocess** (RGB + ImageNet normalization).

## Co rozwiązuje
Pretrained features = lepsza generalizacja na małym datasecie (6k próbek). Mobilenet ma większą receptywność i lepsze low-level features niż PilotNet trenowany od zera.

## Trade-off
**~1.08M params — POWYŻEJ rekomendacji 600k z README PUT**. Może być wolniejszy na JetBot Nano (~15-20 ms inference vs ~5 ms dla PilotNet). Spróbuj jeśli inne modele dryfują w zakrętach.

## Inference (RÓŻNI SIĘ!)
- preprocess: BGR uint8 → **cv2.cvtColor(BGR2RGB)** → /255 → **(x - ImageNet_mean) / ImageNet_std** → CHW → (1, 3, 224, 224) float32
- postprocess: clip ±0.999
