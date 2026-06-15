import cv2
import onnxruntime as rt
import yaml
import numpy as np
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
    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print('No camera')
            break

        result  = ai.predict(image)
        forward = float(result[0])
        left    = float(result[1])


if __name__ == '__main__':
    main()