from PIL import Image

import cv2
import onnxruntime as rt

from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from PUTDriver import PUTDriver, gstreamer_pipeline


class BottomHalfResize:
    """Crop the bottom `fraction` of the image (full width), then resize to `size`×`size`.

    Keeps road/track information and discards uninformative sky/ceiling.
    """
    def __init__(self, fraction: float = 0.5, size: int = 224):
        self.fraction = fraction
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        top = int(h * (1.0 - self.fraction))
        return img.crop((0, top, w, h)).resize((self.size, self.size), Image.BILINEAR)


class DualHeadResNet(nn.Module):
    """ResNet18 backbone with two separate heads:
      - head_val1: regression → tanh output in (-1, 1)
      - head_val2: 3-class classifier → {-1, 0, +1}
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features  # 512
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop fc

        self.head_val1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self.head_val2 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),  # logits for classes: -1, 0, +1
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x).flatten(1)      # (B, 512)
        val1 = self.head_val1(feat).squeeze(1)  # (B,)
        val2_logits = self.head_val2(feat)       # (B, 3)
        return val1, val2_logits
   


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            BottomHalfResize(fraction=0.7, size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        ##TODO: preprocess your input image, remember that img is in BGR channels order
        # Convert BGR to RGB for torchvision transforms
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image for transforms
        img_pil = Image.fromarray(img_rgb)
        
        # Apply transforms
        img_tensor = self.transform(img_pil)
        
        # Add batch dimension and convert to numpy
        img_tensor = img_tensor.unsqueeze(0)
        img_np = img_tensor.numpy().astype(np.float32)
        
        return img_np

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
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() < 1.0
        assert outputs.min() > -1.0

        return outputs


def main():
    with open("config.yml", "r") as stream:
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
