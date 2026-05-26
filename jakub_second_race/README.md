# Jakub — Second Race Solutions

Zestaw 6 modeli ONNX **opset 11** + runtime tunable EMA smoothing.
Każde rozwiązanie self-contained: jedna komenda `python3 bot_driving.py` i jedziemy.

**Empiryczne defaulty** (z 1. wyścigu — działający shufflenet drużynowy):
`max_speed=0.20`, `max_steering=0.7`, `differential=(1.00, 1.00)`, `ema_alpha=0.3`.

## Modele (priorytet testowania na labie)

| # | Model | params | val_loss | preprocess | source |
|---|---|---|---|---|---|
| 1 | **shufflenet_anti_96** ⭐ | 407k | 0.235 | rgb_imagenet 96×96 | NEW (1:1 z działającym shufflenet + nasz anti-bias) |
| 2 | shufflenet_anti_224 | 407k | 0.249 | rgb_imagenet 224×224 | NEW (większy input, bogatsze cechy) |
| 3 | **pilotnet_anti** | 195k | **0.213** | bgr_01 224×224 | NEW (combo wszystkich naszych anti-bias) |
| 4 | mobilenetv3_fl_pretrained | 1.08M | 0.170 | rgb_imagenet 224×224 | RETROFIT z Phase 3 (najlepszy val_loss) |
| 5 | dualhead_class_reg | 240k | 0.211 | bgr_01 224×224 | RETROFIT (klasyfikator left = mniej zygzaków) |
| 6 | pilotnet_fl_shift1 | 195k | 0.216 | bgr_01 224×224 | RETROFIT (label_shift=1) |

Wszystkie modele mają:
- Hflip + target_swap (anti left-bias)
- WeightedRandomSampler na |left| (anti drive-straight)
- Tanh output + clip ±0.999 (strict PUT bounds)
- ONNX opset 11 (PUT contract)
- **EMA smoothing** runtime tunable (anti zygzak)

Modele "anti" mają dodatkowo `label_shift=+2` (~200ms predict-ahead) — adresuje przedwczesny skręt.

## Quick start (lab)

```bash
# 1. Transfer
scp -r first_race_errors/jakub_second_race/ jetbot@<IP>:~/

# 2. Na JetBocie: instalacja onnxruntime (jednorazowo)
ssh jetbot@<IP>
cd ~/jakub_second_race
pip3 install --user -r requirements_jetbot.txt

# JEŚLI 'Illegal instruction (core dumped)' → użyj Jetson Zoo wheel:
#   wget https://nvidia.box.com/shared/static/<...>/onnxruntime_gpu-1.7.0-cp36-cp36m-linux_aarch64.whl
#   pip3 install --user onnxruntime_gpu-1.7.0-cp36-cp36m-linux_aarch64.whl

# 3. Kalibracja differential (jednorazowo per JetBot):
for m in shufflenet_anti_96 shufflenet_anti_224 pilotnet_anti \
         mobilenetv3_fl_pretrained dualhead_class_reg pilotnet_fl_shift1; do
    python3 calibration/setup_differential.py --jetbot 2 --output "$m/config.yml"
    # ↑ zmień "2" na NUMER swojego JetBota (1-4)
done

# Albo interaktywna kalibracja (klawiatura):
python3 calibration/calibrate.py --config shufflenet_anti_96/config.yml

# 4. Test modelu (zacznij od #1):
cd shufflenet_anti_96 && python3 bot_driving.py
```

## Tuning EMA na labie

Jeśli model **zygzakuje na prostych** → mocniejszy smoothing:
```bash
# Edytuj config.yml: inference.ema_alpha 0.3 → 0.2 → 0.1
```

Jeśli model **reaguje za wolno** (over-smoothed) → słabszy smoothing:
```bash
# Edytuj config.yml: inference.ema_alpha 0.3 → 0.5 → 0.8 → 1.0 (raw, brak EMA)
```

Semantyka EMA: `out_t = α·model_out + (1−α)·out_{t-1}`. `α=1.0` = raw, `α→0` = bezwładność.

## Tuning innych knobs

| Problem | Knob | Domyślnie | Tuning |
|---|---|---|---|
| Bot za wolno na prostych | `robot.max_speed` | 0.20 | 0.25, 0.30 |
| Przesadne skręty | `robot.max_steering` | 0.7 | 0.5, 0.4 |
| Bot ucieka w bok | `robot.differential.{left,right}` | (1.0, 1.0) | run `calibrate.py` |
| Zygzak | `inference.ema_alpha` | 0.3 | 0.2, 0.1 |
| Wolna reakcja | `inference.ema_alpha` | 0.3 | 0.5, 0.7, 1.0 |

## Rekomendacja test sequence

1. `shufflenet_anti_96` — najszybszy inference, najbliższy działającemu shufflenet drużynowemu
2. `pilotnet_anti` — najszybszy bez pretrainu, najlepszy nasz val_loss (0.213)
3. `mobilenetv3_fl_pretrained` — największy, ale val_loss 0.170 (najlepszy)
4. `dualhead_class_reg` — klasyfikator left = teoretycznie mniej zygzaków
5. `shufflenet_anti_224` — bogatsze cechy ale wolniejszy
6. `pilotnet_fl_shift1` — fallback (label_shift=1)

## Wymagane biblioteki

- **Host (training)**: patrz [requirements.txt](requirements.txt) (z `resized-no-augmented-jetbot/`)
- **JetBot (runtime)**: patrz [requirements_jetbot.txt](requirements_jetbot.txt) (Python 3.6 + JetPack 4.4)
