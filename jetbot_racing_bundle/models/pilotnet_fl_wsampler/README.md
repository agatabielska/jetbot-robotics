# pilotnet_fl_wsampler

**Wariant**: PilotNet (NVIDIA end-to-end, 1604.07316), target=`forward_left`, anti-bias przez `WeightedRandomSampler`.

## Co rozwiązuje
Główny problem poprzedniej grupy (Kacper-Solution): bez balansu próbek model uczy się ŚREDNIEJ datasetu (forward≈1, left≈0) → "jechaj prosto". Tu binujemy `|left|` na 7 binów i oversampling rzadkich skrętów ~5×.

## Architektura
- 5 Conv (24→36→48→64→64) + BN + ReLU
- 4 Dense (100→50→10→2)
- Output: **tanh** (strict (-1, 1))
- ~260k params

## Trening
`python models/pilotnet_fl_wsampler/train.py`

## Inference (deployment)
- preprocess: BGR uint8 → (1, 3, 224, 224) float32 / 255
- postprocess: clip do ±0.999
