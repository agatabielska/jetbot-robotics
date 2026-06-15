import cv2
import onnxruntime as rt
import yaml
import numpy as np
from collections import deque
from PUTDriver import PUTDriver, gstreamer_pipeline

IMG_SIZE = 96
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']
        self.sess = rt.InferenceSession(
            self.path,
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name  = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        out = np.array(detections).flatten()
        out = np.clip(out, -1.0, 1.0)
        return out.astype(np.float32)

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs     = self.preprocess(img)
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs    = self.postprocess(detections)
        return outputs  # [forward, left]


def sample(buf: deque, d: int, k: int):
    """Average the `k` predictions ending `d` frames in the past.

    d=0 → newest prediction (immediate). d=N → prediction N frames ago.
    Window is clamped while the buffer is still filling up.
    """
    L = len(buf)
    if L == 0:
        return 0.0, 0.0
    end = L - d
    if end < 1:
        end = 1                      # not enough history yet → fall back to oldest
    start = end - k
    if start < 0:
        start = 0
    window = [buf[i] for i in range(start, end)]
    f = sum(w[0] for w in window) / len(window)
    l = sum(w[1] for w in window) / len(window)
    return f, l


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    driver = PUTDriver(config=config)
    ai     = AI(config=config)

    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(flip_method=0, display_width=224, display_height=224),
        cv2.CAP_GSTREAMER
    )

    ret, image = video_capture.read()
    if not ret:
        print('No camera')
        return

    _ = ai.predict(image)

    # ── Adaptive delay buffer ─────────────────────────────────────────────────
    # On a sustained turn the model's lookahead makes the bot steer too early, so
    # we replay the prediction `latency_frames` frames late. On a straight / while
    # straightening out we want an immediate response, so we drop down to
    # `straight_latency_frames`. The big delay engages only after the SAME turn
    # direction has been predicted for `engage_frames` frames (with |left| above
    # `turn_threshold`); a sign flip or a sub-threshold |left| releases it at once.
    rcfg = config['robot']
    latency_frames          = int(rcfg.get('latency_frames', 0))
    straight_latency_frames = int(rcfg.get('straight_latency_frames', 0))
    avg_frames              = max(1, int(rcfg.get('avg_frames', 1)))
    turn_threshold          = float(rcfg.get('turn_threshold', 0.15))
    engage_frames           = max(1, int(rcfg.get('engage_frames', 1)))
    turn_boost_per_frame    = float(rcfg.get('turn_boost_per_frame', 0.0))
    turn_boost_max          = float(rcfg.get('turn_boost_max', 1.0))

    maxlen = max(latency_frames, straight_latency_frames) + avg_frames + 1
    buffer = deque(maxlen=maxlen)

    turn_dir        = 0   # -1 / 0 / +1 : direction of the currently held turn
    same_turn_count = 0   # consecutive frames the same turn has been predicted

    print(f'Adaptive buffer: turn={latency_frames}f  straight={straight_latency_frames}f  '
          f'avg={avg_frames}  thr={turn_threshold}  engage={engage_frames}  '
          f'boost={turn_boost_per_frame}/f<={turn_boost_max}')
    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    boost         = 1.0
    while True:
        engaged_flag = same_turn_count >= engage_frames and turn_dir != 0
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}\t'
              f'turn={turn_dir:+d} cnt={same_turn_count} buf={"ON " if engaged_flag else "off"} '
              f'x{boost:.2f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print('No camera')
            break

        result       = ai.predict(image)
        cur_f, cur_l = float(result[0]), float(result[1])
        buffer.append((cur_f, cur_l))

        # Turn detection on the fresh (un-delayed) prediction.
        sign = 0
        if abs(cur_l) >= turn_threshold:
            sign = 1 if cur_l > 0 else -1

        if sign != 0 and sign == turn_dir:
            same_turn_count += 1
        else:
            turn_dir        = sign
            same_turn_count = 1 if sign != 0 else 0

        engaged = sign != 0 and same_turn_count >= engage_frames
        delay   = latency_frames if engaged else straight_latency_frames
        forward, left = sample(buffer, delay, avg_frames)

        # Progressive turn boost: the longer the same turn is held in a row, the
        # harder we steer (acts like an integral term for long / tight curves).
        boost = 1.0
        if sign != 0 and turn_boost_per_frame > 0.0:
            boost = 1.0 + turn_boost_per_frame * max(0, same_turn_count - 1)
            boost = min(boost, turn_boost_max)
            left  = float(np.clip(left * boost, -1.0, 1.0))


if __name__ == '__main__':
    main()
