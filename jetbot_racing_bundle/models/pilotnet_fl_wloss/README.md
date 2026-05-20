# pilotnet_fl_wloss

**Wariant**: PilotNet, `forward_left`, anti-bias przez **per-dim weighted Huber loss** (weights=[1.0, 3.0]).

## Co rozwiązuje
Alternative do `wsampler`. Gradient na `left` jest 3× większy niż na `forward`, więc model uczy się steeringu nawet jeśli próbki są zdominowane przez `forward=1`.

## Inference
- preprocess: BGR uint8 → (1, 3, 224, 224) float32 / 255
- postprocess: clip ±0.999
