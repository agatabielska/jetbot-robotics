# comp-vision-olek

CV-based lane following for the JetBot. Uses classical computer vision (binary
thresholding) to locate the road centre in each camera frame, then feeds the
detected coordinates into a small trained MLP to produce drive commands.

## Pipeline overview

```
camera frame
     │
     ▼
binary threshold          ← binary_threshold_test.py / preprocess_dataset.py
     │
     ▼
largest white component
centre (cx, cy)
     │
     ▼
normalise: cx/224, cy/224
     │
     ▼
ONNX MLP model            ← cv_checkpoints/jetbot_cv_mlp.onnx
     │
     ▼
(forward, turn) ∈ [-1, 1]
     │
     ▼
PUTDriver.update(forward, turn)
```

## Files

| File | Purpose |
|---|---|
| `binary_threshold_test.py` | Interactive visualiser — shows binarised frames with detected centre |
| `preprocess_dataset.py` | Offline: applies the CV pipeline to every dataset image and writes `dataset_preprocessed/` |
| `cv_models.py` | MLP architectures (`CoordMLP`, `CoordMLPDeep`) |
| `cv_dataset.py` | Dataset loader for training (reads `dataset_preprocessed/` CSVs) |
| `cv_train.py` | Training script — exports ONNX on completion |
| `cv_test_onnx.py` | Evaluates ONNX model using pre-extracted coordinates from `dataset_preprocessed/` |
| `cv_test_images.py` | **End-to-end** evaluation — runs the full CV pipeline on raw images, comparable to `resized-jetbot/test_onnx.py` |

---

## Deploying on the real robot

### What to run for every camera frame

For each frame captured from the JetBot camera you must:

1. **Apply binary thresholding** using exactly the same parameters as during
   preprocessing (`THRESHOLD_VALUE = 110`):
   - Convert to grayscale
   - Binary threshold at 110
   - Zero out everything above the bottom third of the image (remove sky/background)
   - Zero out the left and right thirds (keep only the central lane region)
   - Morphological close → open with a 5×5 kernel to clean noise

2. **Find the largest white connected component** and read its centroid `(cx, cy)`.
   If no component is found, the robot cannot determine its position — hold the
   last known command or stop.

3. **Normalise the coordinates**:
   ```
   cx_norm = cx / 224
   cy_norm = cy / 224
   ```

4. **Run the ONNX model** with `[cx_norm, cy_norm]` as input → get `[forward, turn]`.

5. **Drive**: pass `forward` and `turn` to `PUTDriver.update()`.

### Minimal inference code

#### (I dont know if this code will work)
```python
import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE        = 224
THRESHOLD_VALUE = 110

sess       = ort.InferenceSession("cv_checkpoints/jetbot_cv_mlp.onnx",
                                   providers=["TensorrtExecutionProvider",
                                              "CUDAExecutionProvider",
                                              "CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

forward, turn = 0.0, 0.0

while True:
    ret, frame = video_capture.read()       # BGR frame from GStreamer camera
    if not ret:
        break

    # 1. Binary threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    binary[: (h * 2) // 3, :] = 0          # keep bottom third only
    binary[:, : w // 3]        = 0          # keep centre horizontal third only
    binary[:, (w * 2) // 3 :]  = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # 2. Detect centre of largest component
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if n_labels > 1:                        # at least one non-background component
        best  = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx    = int(round(centroids[best + 1][0]))
        cy    = int(round(centroids[best + 1][1]))

        # 3. Normalise
        coords = np.array([[cx / IMG_SIZE, cy / IMG_SIZE]], dtype=np.float32)

        # 4. Infer
        output       = sess.run(None, {input_name: coords})[0].flatten()
        forward, turn = float(output[0]), float(output[1])

    # 5. Drive (reuses last command if no detection this frame)
    driver.update(forward, turn)
```

### Things to keep consistent with training

| Parameter | Value | Where it is set |
|---|---|---|
| Image size assumed by normalisation | 224 × 224 | `cv_models.py → IMG_SIZE` |
| Binary threshold value | 110 | `preprocess_dataset.py → THRESHOLD_VALUE` |
| ROI: vertical | bottom third | `apply_binary_threshold()` |
| ROI: horizontal | centre third | `apply_binary_threshold()` |
| Morphological kernel | 5 × 5 rect | `apply_binary_threshold()` |

If you change the threshold or ROI, you must re-run `preprocess_dataset.py` and
retrain the model — the coordinate distribution seen at training time must match
what the robot produces at inference time.

---

## Training

```bash
# from the resized-jetbot/ directory (where the uv environment lives)

# 1. preprocess the raw dataset (only needed once)
uv run python ../comp-vision-olek/preprocess_dataset.py

# 2. train
uv run python ../comp-vision-olek/cv_train.py --epochs 150

# 3. evaluate end-to-end on raw test images
uv run python ../comp-vision-olek/cv_test_images.py --onnx ../comp-vision-olek/cv_checkpoints/jetbot_cv_mlp.onnx --all
```
