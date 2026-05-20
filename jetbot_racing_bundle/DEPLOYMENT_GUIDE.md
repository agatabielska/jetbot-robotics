# DEPLOYMENT GUIDE — JetBot Racing (Phase 3)

Krok po kroku: Ubuntu host → JetBot → kalibracja → test wszystkich 6 modeli.
Wszystkie komendy są **copy-paste ready**.

---

## 0. Prerequisites

**Na laptopie/desktopie (Ubuntu)**:
- Docker zainstalowany (`docker --version`)
- Docker buildx (`docker buildx version`) — dla cross-build ARM64
- SSH client (`ssh -V`)
- Image jetbot-racing.tar.gz **już pobrany na dysk** (lub buildowany lokalnie — sekcja 1)

**Na JetBocie**:
- Docker zainstalowany (jest pre-installed w JetPack ≥ 4.3)
- `nvidia-container-toolkit` (jest pre-installed)
- SSH dostęp (`ssh jetbot@<IP>`, hasło: `jetbot`)
- Kamera CSI podłączona

---

## 1. Build image (na hoście — JEDEN RAZ)

> Jeśli już masz `jetbot-racing.tar.gz` na dysku — **pomiń tę sekcję**, przejdź do sekcji 2.

```bash
# Wejdź do katalogu projektu Phase 3
cd /path/to/JetBot/model_training_phase3

# Inicjalizuj buildx (jednorazowo)
docker buildx create --use --name jetbot-builder
docker buildx inspect --bootstrap

# Zbuduj image dla ARM64
./scripts/build_docker.sh

# Zapisz image do pliku .tar.gz (gotowy do transferu)
docker save jetbot-racing:phase3 | gzip > jetbot-racing.tar.gz
ls -lh jetbot-racing.tar.gz   # spodziewaj się ~1-2 GB
```

---

## 2. Transfer image na JetBota

```bash
# Z laptopa do JetBota (zastąp <JETBOT_IP> rzeczywistym adresem IP)
JETBOT_IP=192.168.0.42                     # zmień!

# Transfer (~1-2 min na WiFi)
scp jetbot-racing.tar.gz jetbot@$JETBOT_IP:~/
# Hasło: jetbot
```

---

## 3. SSH do JetBota i załaduj image

```bash
ssh jetbot@$JETBOT_IP                       # hasło: jetbot

# (już na JetBocie)
cd ~
ls -lh jetbot-racing.tar.gz                 # potwierdź że plik jest

# Załaduj image do lokalnego docker (~30s)
zcat jetbot-racing.tar.gz | docker load

# Sprawdź że image jest
docker images | grep jetbot-racing
# Powinno pokazać: jetbot-racing  phase3  <hash>  <date>  ~2GB
```

---

## 4. Zatrzymaj inne kontenery (opcjonalnie, jeśli ktoś coś odpalił)

```bash
# Lista uruchomionych kontenerów
docker ps

# Zatrzymaj WSZYSTKIE działające kontenery (na JetBocie i tak nic poważnego nie powinno działać)
docker stop $(docker ps -q) 2>/dev/null || echo "Nothing to stop"

# Usuń stopped containers (cleanup miejsca)
docker container prune -f

# (Opcjonalnie, jeśli na JetBocie jest mało miejsca: usuń stare nie używane obrazy)
# docker image prune -a -f  # OSTROŻNIE: usunie WSZYSTKIE nieużywane obrazy
```

---

## 5. Quick differential setup (JEDNORAZOWO per JetBot)

JetBot wymaga kalibracji kół, bo silniki mają tolerancję — bez tego dryfuje.

**Opcja A**: znasz numer swojego JetBota (1-4) — użyj presetu:
```bash
# Utwórz lokalnie config.yml (na JetBocie)
docker run --rm -it \
    -v $HOME/config.yml:/workspace/config.yml \
    jetbot-racing:phase3 setup
# → Wybierz numer JetBota (1-4) z prompta, np. "2"
# → Zapisuje ~/config.yml z poprawnymi differential.left/right
```

**Opcja B**: nie znasz numeru lub wartości z tabeli nie są aktualne — przejdź od razu do kalibracji (sekcja 6).

**Opcja C**: ustaw konkretne wartości CLI:
```bash
docker run --rm \
    -v $HOME/config.yml:/workspace/config.yml \
    jetbot-racing:phase3 setup --left 0.92 --right 1.0
# (Z poziomu run.sh możliwe jest nadpisanie argumentów — patrz `run.sh`.)
```

Po setupie sprawdź:
```bash
cat ~/config.yml
# Powinno pokazać robot.differential.{left,right} = wartości z presetu
```

---

## 6. Kalibracja precyzyjna (zalecane PRZED pierwszym wyścigiem)

Postaw JetBota na płaskiej podłodze, **przed nim ~1m wolnej przestrzeni**.

```bash
docker run --rm -it --runtime nvidia \
    -v $HOME/config.yml:/workspace/config.yml \
    jetbot-racing:phase3 calibrate
```

W kontenerze — klawiatura:

| Key | Action |
|---|---|
| `w` | jedź przy `forward=0.5` |
| `s` | stop |
| `q` / `e` | -/+ differential.left (krok 0.05) |
| `a` / `d` | -/+ differential.right |
| `r` | reset do (1.0, 1.0) |
| `p` | zapisz do config.yml |
| `x` | wyjdź |

**Procedura**:
1. `w` → JetBot jedzie. Obserwuj drift.
2. Bot dryfuje w lewo? → `e` (zwiększ left). Lub `a` (zmniejsz right).
3. Bot dryfuje w prawo? → `q` (zmniejsz left). Lub `d` (zwiększ right).
4. Powtarzaj aż JetBot jedzie prosto ~1m.
5. `p` → save. `x` → exit.

Pełna instrukcja: [deployment/CALIBRATION.md](deployment/CALIBRATION.md) (skopiowana do `/workspace/CALIBRATION.md` w kontenerze).

---

## 7. Test każdego modelu po kolei

**Wariant A** (interaktywny test jeden-po-jednym, w pętli):

```bash
for MODEL in pilotnet_fl_wsampler pilotnet_fl_wloss pilotnet_fl_shift1 \
             pilotnet_motor_wsampler dualhead_class_reg mobilenetv3_fl_pretrained; do
    echo ""
    echo "============================================="
    echo "=== Testing: $MODEL"
    echo "============================================="
    docker run --rm -it --runtime nvidia --device /dev/video0 \
        -v $HOME/config.yml:/workspace/config.yml \
        jetbot-racing:phase3 $MODEL

    read -p "Naciśnij Enter aby przejść do kolejnego modelu (Ctrl+C aby przerwać)..."
done
```

**Wariant B** (pojedynczy model — szybki test):

```bash
# Najlepszy baseline — START TUTAJ
docker run --rm -it --runtime nvidia --device /dev/video0 \
    -v $HOME/config.yml:/workspace/config.yml \
    jetbot-racing:phase3 pilotnet_fl_wsampler

# Po `Press Enter to start` JetBot zaczyna jeździć autonomicznie.
# Ctrl+C aby zatrzymać (silniki zostaną wyłączone w finally{}).
```

**Inne dostępne modele** (każdy jako ostatni argument `docker run`):
- `pilotnet_fl_wsampler` ⭐ (REKOMENDOWANY)
- `pilotnet_fl_wloss`
- `pilotnet_fl_shift1`
- `pilotnet_motor_wsampler`
- `dualhead_class_reg`
- `mobilenetv3_fl_pretrained`

Pełen opis: [MODELS.md](MODELS.md).

---

## 8. Cleanup po zawodach

```bash
# Zatrzymaj WSZYSTKIE kontenery z naszego obrazu
docker stop $(docker ps -q --filter ancestor=jetbot-racing:phase3) 2>/dev/null || true

# (Opcjonalnie) usuń image z dysku JetBota
# docker rmi jetbot-racing:phase3
# rm ~/jetbot-racing.tar.gz
```

---

## 9. Debug shell (jeśli coś nie działa)

```bash
# Wejdź do kontenera bash (nie odpalaj modelu)
docker run --rm -it --runtime nvidia --device /dev/video0 \
    -v $HOME/config.yml:/workspace/config.yml \
    --entrypoint bash \
    jetbot-racing:phase3

# (w środku)
ls /workspace/models/                              # 6 folderów modeli
python3 -c "import onnxruntime; print(onnxruntime.__version__)"  # 1.10.0
python3 -c "import jetbot; print('jetbot OK')"     # preinstalled
ls /workspace/models/pilotnet_fl_wsampler/         # bot_driving + best.onnx + PUTDriver + config.yml + preprocess + postprocess
```

---

## 10. Troubleshooting

### Problem: `Failed to open device /dev/video0`
- Kamera niewidoczna. Sprawdź `ls -la /dev/video*` na JetBocie (bez kontenera). Spróbuj re-podłączyć kamerę CSI lub `sudo modprobe v4l2_common`.

### Problem: `Cannot import 'jetbot'`
- Kontener nie ma `jetbot` library. Build base image był zły. Re-buduj z poprawnym base `nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3`.

### Problem: Bot non-stop skręca w jedną stronę przy `pilotnet_fl_wsampler`
- Najpierw sprawdź kalibrację — sekcja 6. Jeśli wciąż — zmień model na inny (patrz `MODELS.md` "Common failure modes").

### Problem: Inference za wolny (bot reaguje opóźnione)
- Sprawdź jakie providers wybrał:
  ```bash
  docker run --rm -it --runtime nvidia jetbot-racing:phase3 pilotnet_fl_wsampler
  # Powinno wypluć: [AI] Providers (priority): ['CPUExecutionProvider'] (lub CUDA/TRT jeśli dostępne)
  ```
- Na Jetson Nano CPU latency ~5-10ms — wystarczająca dla 10Hz control loop.
- Jeśli używasz MobileNetV3 — to model jest 3× większy, normalne że wolniej. Przejdź na PilotNet.

### Problem: Image za duży do transferu
- `jetbot-racing.tar.gz` ~1-2 GB to spodziewane. Użyj USB stick jeśli WiFi za wolny.
- Alternatywa: zbuduj na samym JetBocie (`cd ~ && git clone <repo> && cd model_training_phase3 && docker build -f deployment/Dockerfile .`) — zajmie znacznie dłużej ale unikniesz transferu.

### Problem: `External data path validation failed for initializer`
- Brakuje pliku `best.onnx.data` obok `best.onnx`. Wszystkie nasze ONNX-y mają sidecar — upewnij się że oba pliki są w `/workspace/models/<name>/`. Bug naszego buildu jeśli brakuje.

### Problem: `No space left on device` na JetBocie
- Wyczyść stare obrazy: `docker image prune -a -f` (OSTROŻNIE — usuwa wszystkie nieużywane).

---

## Quick reference

```bash
# Setup (jednorazowo):
zcat jetbot-racing.tar.gz | docker load
docker run --rm -it -v $HOME/config.yml:/workspace/config.yml jetbot-racing:phase3 setup
docker run --rm -it --runtime nvidia -v $HOME/config.yml:/workspace/config.yml jetbot-racing:phase3 calibrate

# Test modelu:
docker run --rm -it --runtime nvidia --device /dev/video0 \
    -v $HOME/config.yml:/workspace/config.yml \
    jetbot-racing:phase3 pilotnet_fl_wsampler

# Cleanup:
docker stop $(docker ps -q --filter ancestor=jetbot-racing:phase3) 2>/dev/null
```
