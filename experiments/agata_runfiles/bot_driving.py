from PIL import Image

import cv2
import onnxruntime as rt

from pathlib import Path
import yaml
import numpy as np

from PUTDriver import PUTDriver, gstreamer_pipeline


class BottomHalfResize:
    """Crop the bottom `fraction` of the image (full width), then resize to `size`×`size`.

    Keeps road/track information and discards uninformative sky/ceiling.
    """
    def __init__(self, fraction: float = 0.5, size: int = 224):
        self.fraction = fraction
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        top = int(h * (1.0 - self.fraction))
        cropped = img[top:h, 0:w]
        resized = cv2.resize(cropped, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return resized


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.output_names = [o.name for o in self.sess.get_outputs()]

        for o in self.sess.get_outputs():
            print(o.name, o.shape, o.type)
        self.input_name = self.sess.get_inputs()[0].name

        # Store normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Initialize transform pipeline
        self.bottom_half_resize = BottomHalfResize(fraction=0.7, size=224)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
                # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 256 (bilinear)
        img_256 = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Apply BottomHalfResize
        img_cropped_resized = self.bottom_half_resize(img_256)
        
        # Convert to float32 and normalize to [0, 1]
        img_float = img_cropped_resized.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW (for ONNX model)
        #TODO check if this line is important
        img_chw = np.transpose(img_float, (2, 0, 1))
        
        # Normalize with mean and std
        img_normalized = (img_chw - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]
        
        # Add batch dimension
        img_tensor = np.expand_dims(img_normalized, axis=0)
        
        return img_tensor.astype(np.float32)

    def postprocess(self, detections: tuple) -> np.ndarray:
        """Process dual head outputs: (val1, val2_logits) → (forward, left)
        
        Args:
            detections: tuple of (val1, val2_logits)
                - val1: shape (1,), values in (-1, 1)
                - val2_logits: shape (1, 3), raw logits for 3 classes: -1, 0, +1
        
        Returns:
            outputs: shape (2,), values in (-1, 1) for [forward, left]
        """
        val1, val2_logits = detections
        
        # val1 is already in (-1, 1) from tanh - use as forward
        forward = val1[0] if isinstance(val1, np.ndarray) else float(val1)
        
        # Convert val2_logits to class index (0, 1, 2) → (-1, 0, +1)
        val2_class_idx = np.argmax(val2_logits[0])  # Get predicted class
        left = float(val2_class_idx - 1)  # Map 0→-1, 1→0, 2→+1
        
        return np.array([forward, left], dtype=np.float32)

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)
        
        detections = self.sess.run(self.output_names, {self.input_name: inputs})
        outputs = self.postprocess(detections)
        outputs[0] = max(-1.0, min(outputs[0], 1.0))
        outputs[1] = max(-1.0, min(outputs[1], 1.0))
        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() <= 1.0
        assert outputs.min() >= -1.0

        return outputs


def main():
    with open("agata_runfiles/config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print(f'No camera')
        return
    
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break
        forward, left = ai.predict(image)


if __name__ == '__main__':
    main()
jetbot@jetso
