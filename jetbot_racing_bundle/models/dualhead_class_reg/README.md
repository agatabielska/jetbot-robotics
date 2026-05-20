# dualhead_class_reg

**Wariant**: PilotNet backbone z **dwoma głowami** (klasyfikator + regresor):
- Head A: **classifier** dla `left` — 7 binów discretization, CrossEntropy
- Head B: **regressor** dla `forward` — smooth L1 loss

## Co rozwiązuje
Najbardziej radykalna odpowiedź na "drive-straight bias" — klasyfikacja na binach Encoder uczy się DECISION (skręcaj w lewo / prosto / w prawo / mocno w prawo) zamiast średniej. Regresja `forward` zostaje osobno, więc model nie miesza tych dwóch sygnałów.

## ONNX export
W trybie treningu model emituje `(left_logits, forward_pred)` — TUPLE. Dla ONNX export ustawiamy `model.export_mode = True` (patrz `phase3_common/export_onnx.py`), co konwertuje:
1. `softmax(left_logits)` → softmax-weighted average bin center → smooth left value ∈ (-1, 1)
2. concat z `forward_pred` → finalny `(B, 2)` tensor

Dzięki temu deployment używa standardowego forward_left postprocess (`clip ±0.999`).

## Inference
- preprocess: BGR uint8 → (1, 3, 224, 224) float32 / 255
- postprocess: clip ±0.999 (wewnętrznie ONNX już zwraca w (-1, 1))
