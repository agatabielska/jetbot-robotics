# pilotnet_motor_wsampler

**Wariant**: PilotNet, target=**`motor_lr`** (left/right motor speeds zamiast forward_left), WeightedRandomSampler na binach `|m_l - m_r|` (turn magnitude).

## Co rozwiązuje
Phase 2 pokazał że target `motor_lr` ma lepiej zbalansowany rozkład saturacji (33% vs 44.7% dla forward_left). Tu sprawdzamy empirycznie czy training jest stabilniejszy.

## Inference
- preprocess: BGR uint8 → (1, 3, 224, 224) float32 / 255
- postprocess: **konwertuje (m_l, m_r) → (forward, left)** przez inverse PUTDriver mapping, potem clip ±0.999. To kluczowa różnica vs forward_left modele — `bot_driving.py` zawsze dostaje (forward, left) zgodnie z PUT contract.
