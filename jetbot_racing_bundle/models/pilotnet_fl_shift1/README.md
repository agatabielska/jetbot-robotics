# pilotnet_fl_shift1

**Wariant**: PilotNet, `forward_left`, `label_shift=+1` (predykcja sterowania ~200 ms w przód).

## Co rozwiązuje
README PUT explicit tip: "how about doing some data shift and predicting 1/2 of the control values ahead to deal with latency?". Natural shift z `user_driving.py` to ~100 ms; dodatkowy +1 daje łącznie ~200 ms. Lepsza odporność na inference latency loop.

## Inference
- preprocess: BGR uint8 → (1, 3, 224, 224) float32 / 255
- postprocess: clip ±0.999
