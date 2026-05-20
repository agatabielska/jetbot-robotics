# MODELS — Phase 3 Variants Catalog

Sześć wariantów modeli wytrenowanych na datasecie z `dataset_analysis_phase1/outputs/tables/split_B_holdout.csv` (6109/1475 train/val, hold-out per session). Każdy adresuje WSKAZANE w `first_race_errors/REPORT.md` problemy: **left-bias** (brak hflip) i **drive-straight** (model uczy się ŚREDNIEJ przy braku balansu).

Wszystkie warianty mają **gwarantowane**:
- Hflip + target swap w augmentacjach (Phase 2 — adresuje left-bias)
- Strict bounds clipping ±0.999 w postprocess (kontrakt PUT)
- ONNX opset 11, input `(1, 3, 224, 224) float32`, output `(2,) float32`
- Anti-bias mechanizm specyficzny dla wariantu

---

## Tabela porównawcza

| # | Model | Params | val_loss\* | Target | Anti-bias | Preprocess | Architektura |
|---|---|---|---|---|---|---|---|
| 1 | **pilotnet_fl_wsampler** ⭐ | 240k | (po retrain) | forward_left | **WeightedRandomSampler** (7 binów `\|left\|`) | bgr_01 (/255) | PilotNet (5 conv + 4 dense, tanh) |
| 2 | pilotnet_fl_wloss | 240k | (po retrain) | forward_left | **Per-dim weighted Huber** w=[1.0, 3.0] | bgr_01 | PilotNet (j.w.) |
| 3 | pilotnet_fl_shift1 | 240k | (po retrain) | forward_left | wsampler + **label_shift=+1** (predict +200ms) | bgr_01 | PilotNet |
| 4 | pilotnet_motor_wsampler | 240k | (po retrain) | **motor_lr** | wsampler na `\|m_l - m_r\|` | bgr_01 | PilotNet |
| 5 | dualhead_class_reg | 240k | (po retrain) | forward_left | **Klasyfikator** 7-bin `left` + **regresor** `forward` | bgr_01 | PilotNet 2-head |
| 6 | mobilenetv3_fl_pretrained | 1.08M | (po retrain) | forward_left | wsampler | **rgb_imagenet** | MobileNetV3-Small + tanh head |

\* val_loss tylko porównywalny w obrębie tego samego `target_type`. motor_lr działa na innej skali.

⭐ **Rekomendowany baseline** (wsampler bezpośrednio adresuje "drive-straight" bug Kacpra).

---

## Strategia użycia na zawodach (priorytet)

1. **Start**: `pilotnet_fl_wsampler` (najlepszy mix prostoty, szybkości inference, anti-bias)
2. Jeśli dryfuje w bok → `pilotnet_fl_wloss` (alternatywny anti-bias)
3. Jeśli reaguje za późno (ciągle wpada w zakręty) → `pilotnet_fl_shift1` (+200ms forecast)
4. Jeśli wszystkie PilotNet'y mają drive-straight bias → `dualhead_class_reg` (klasyfikator)
5. Eksperymentalnie: `pilotnet_motor_wsampler` (motor_lr — alternatywny target)
6. Ostatnia deska ratunku: `mobilenetv3_fl_pretrained` (większy + pretrained features; **WOLNIEJSZY** inference ~15-20ms vs ~5ms)

---

## Per-model szczegóły

### 1. pilotnet_fl_wsampler ⭐

**Architektura**: PilotNet ([NVIDIA 1604.07316](https://arxiv.org/pdf/1604.07316.pdf))
- 5 Conv (24 → 36 → 48 → 64 → 64), kernel 5×5/3×3, BatchNorm + ReLU
- AdaptiveAvgPool2d(4) → 1024 features
- 4 Dense (100 → 50 → 10 → 2)
- Output: `tanh` → zawsze ∈ (-1, 1)

**Anti-bias**: `WeightedRandomSampler` na binach `|left|` (k=7). Rzadkie zakręty oversamplowane ~5×. Każdy batch ma ~równe pokrycie "prosto" i "ostro w bok".

**Kiedy używać**: zawsze jako pierwszy. Solidny default.

**Kiedy unikać**: gdy widzisz że bot REAGUJE na obrazy ale dryfuje (np. zawsze 30° w lewo) → spróbuj inne warianty.

**Kontrakt I/O**:
- preprocess: BGR uint8 224×224 → `(1, 3, 224, 224) float32 [0, 1]`
- postprocess: ONNX out → clip `[-0.999, 0.999]` → `(forward, left)`

### 2. pilotnet_fl_wloss

**Architektura**: identyczna jak #1.

**Anti-bias**: zamiast resamplowania batchu — **per-dim weighted Huber loss**:
```
loss = mean(w_forward · huber(pred_f, gt_f) + w_left · huber(pred_l, gt_l))
       weights = [1.0, 3.0]
```
Gradient na `left` 3× silniejszy. Model dostaje wszystkie próbki ze standardowym rozkładem, ale steering "boli" 3× bardziej. Mniej hacky niż wsampler.

**Kiedy używać**: alternatywa do #1 jeśli widzisz że wsampler zawodzi.

**Kiedy unikać**: jeśli predykcje są zbyt jittery (wzmocniony gradient = większe oscylacje).

**Kontrakt**: identyczny do #1.

### 3. pilotnet_fl_shift1

**Architektura**: identyczna do #1.

**Modyfikacja danych**: `label_shift=+1` — etykieta `(forward, left)` z klatki `t+1` przypisana do klatki `t`. Razem z naturalnym ~100ms shift z `user_driving.py:prev_image` mechanizmu → łącznie **~200ms predict-ahead**. README PUT explicit suggests this: *"how about doing some data shift and predicting 1/2 of the control values ahead to deal with latency?"*.

**Kiedy używać**: jeśli widzisz że bot ZA PÓŹNO reaguje na zakręty (wpada w nie, krawędziuje tor). Dodatkowy lookahead pozwala wcześniej zacząć skręt.

**Kiedy unikać**: jeśli bot skręca PRZEDWCZEŚNIE (model już za bardzo przewiduje → tnie zakręty).

**Kontrakt**: identyczny do #1.

### 4. pilotnet_motor_wsampler

**Architektura**: identyczna do #1, ale **target = (motor_left, motor_right)** zamiast (forward, left). Konwersja z `forward_left_to_motors_raw` (Phase 2 `common/targets.py`).

**Anti-bias**: WeightedRandomSampler na `|motor_l - motor_r|` (turn magnitude w przestrzeni motorów).

**Postprocess (UWAGA)**: ONNX zwraca `(m_l, m_r)`, postprocess konwertuje przez **inverse PUTDriver mapping** → `(forward, left)`, potem clip. Driver dostaje `(forward, left)` jak każdy inny model — kontrakt PUT zachowany.

**Kiedy używać**: gdy forward_left modele dryfują — może motor space ma lepiej zbalansowany rozkład (Phase 2: ~33% saturation vs ~44.7% forward).

**Kiedy unikać**: bez specjalnego powodu. To eksperymentalna ścieżka.

**Kontrakt I/O** (wyjątek!): preprocess identyczny, postprocess **konwertuje motor_lr → forward_left**. Model ONNX widzi 2 wyjścia w (-1.5, 1) (motor_lr może przekroczyć 1) — postprocess clip ratuje strict bound.

### 5. dualhead_class_reg

**Architektura**: PilotNet backbone (shared conv + 2-layer dense) → **dwie głowy**:
- Head A: `Linear(50 → 7)` — klasyfikator dla `left` w 7 binach (CrossEntropy)
- Head B: `Linear(50 → 10) → Linear(10 → 1) → tanh` — regresor dla `forward` (smooth_l1)

**Loss**: `CE(left_logits, left_bin_idx) + smooth_l1(forward_pred, forward_gt)`.

**ONNX export trick**: w trybie treningowym model zwraca tuple. Przy export `model.export_mode = True` aktywuje wrapper:
```
softmax(left_logits) · bin_centers → smooth left ∈ (-1, 1)
concat z forward_pred → (B, 2)
```
Deployment widzi standardowy `(B, 2)` ONNX — kompatybilny z PUT.

**Kiedy używać**: najsilniejsza obrona przed drive-straight bias. Model uczy się DECYZJI ("skręcaj w lewo / prosto / w prawo / mocno w prawo"), nie regresji do średniej.

**Kiedy unikać**: jeśli skręty są zbyt "schodkowane" (binowanie powoduje że bot porusza się dyskretnie zamiast płynnie) — wtedy `pilotnet_fl_wloss` może być lepsze.

**Kontrakt**: identyczny do #1 (wrapper export to ukrywa).

### 6. mobilenetv3_fl_pretrained

**Architektura**: `torchvision.models.mobilenet_v3_small` z weights `IMAGENET1K_V1`. Classifier wymieniony na:
```
Linear(in → 256) → Hardswish → Dropout(0.2) → Linear(256 → 2) → tanh
```
~1.08M params (3× większy niż PilotNet, **POWYŻEJ rekomendacji ~600k z PUT**).

**Anti-bias**: WeightedRandomSampler.

**Preprocess INNY**: `rgb_imagenet` — BGR→RGB konwersja + ImageNet mean/std normalization. To wymóg pretrained weights ImageNet.

**Kiedy używać**: gdy potrzebujesz lepszej generalizacji (pretrained ImageNet features). Trade-off: wolniejszy inference.

**Kiedy unikać**: jeśli inference latency jest krytyczne — `~15-20ms vs ~5ms` na Jetson Nano CPU.

**Kontrakt I/O** (wyjątek!): preprocess **inny od pozostałych** (RGB, normalized). Postprocess standard clip.

---

## Common failure modes & remedies

| Symptom | Możliwa przyczyna | Spróbuj |
|---|---|---|
| Bot jedzie prosto bez skrętów | drive-straight bias (avg fit) | Przejdź na `dualhead_class_reg` lub `pilotnet_fl_wloss` |
| Bot non-stop skręca w jedną stronę | calibration błędna LUB model left/right-bias | Najpierw uruchom `setup` lub `calibrate`. Jeśli wciąż — zmień model. |
| Bot reaguje za późno (krawędziuje tor) | lookahead za krótki | `pilotnet_fl_shift1` |
| Bot przedwcześnie skręca | za duży lookahead | `pilotnet_fl_wsampler` (default) |
| Predykcje jittery (oscylacje) | over-amplified gradient | Z `_wloss` na `_wsampler` |
| Inference za wolny | model za duży | NIE używaj `mobilenetv3` — wybierz PilotNet variant |
| `assert outputs.max() < 1.0` fail | postprocess clip nie zadziałał | Bug w naszym kodzie — report; tymczasowo zmień model |

---

## Reference

- Phase 1 wnioski: [../dataset_analysis_phase1/CLAUDE.md](../dataset_analysis_phase1/CLAUDE.md)
- Phase 2 wnioski: [../data_preparation_phase2/CLAUDE.md](../data_preparation_phase2/CLAUDE.md)
- Analiza porażek pierwszego wyścigu: [../first_race_errors/REPORT.md](../first_race_errors/REPORT.md)
- Per-model README: `models/<name>/README.md`
- Training results: [outputs/report.md](outputs/report.md), [outputs/tables/val_metrics_all.csv](outputs/tables/val_metrics_all.csv)
