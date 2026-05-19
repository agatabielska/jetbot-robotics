"""Minimal JetBot driver + camera pipeline. Standalone — only depends on
the `jetbot` SDK (preinstalled on Waveshare JetBot image) and pyyaml.

Convention:
  - left > 0 = turn LEFT (left wheel slows)
  - left < 0 = turn RIGHT (right wheel slows)
  - forward > 0 = drive forward
"""

import yaml
from jetbot import Robot


def gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 224,
    display_height: int = 224,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, "
        f"height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


class Driver:
    def __init__(self, config: dict):
        self.robot = Robot()
        r = config["robot"]
        self.max_speed = r["max_speed"]
        self.max_steering = r["max_steering"]
        self.left_c = r["differential"]["left"]
        self.right_c = r["differential"]["right"]

    def update(self, forward: float, left: float) -> None:
        left_speed = forward * self.left_c
        right_speed = forward * self.right_c
        if left > 0:
            left_speed -= left * self.max_steering
        elif left < 0:
            right_speed += left * self.max_steering
        self.robot.set_motors(
            left_speed=left_speed * self.max_speed,
            right_speed=right_speed * self.max_speed,
        )

    def stop(self) -> None:
        self.robot.set_motors(0.0, 0.0)


def load_config(path: str = "config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
