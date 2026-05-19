"""Motor sign-convention verification. Run with the bot ON BLOCKS so the
wheels do not touch the ground.

Expected behavior:
  [2/4] left=+0.5  -> bot would turn LEFT  (left wheel slows down)
  [3/4] left=-0.5  -> bot would turn RIGHT (right wheel slows down)

If wheels do the opposite, flip the sign in drive.py:
    driver.update(forward=forward, left=-left_signal)
"""

import os
import time

from driver import Driver, load_config

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = load_config(os.path.join(HERE, "config.yml"))
    d = Driver(cfg)

    print("[1/4] forward only (both wheels same speed)")
    d.update(forward=0.3, left=0.0)
    time.sleep(2)
    d.stop()
    time.sleep(0.5)

    print("[2/4] left=+0.5  -> bot SHOULD turn LEFT (left wheel slows)")
    d.update(forward=0.3, left=+0.5)
    time.sleep(2)
    d.stop()
    time.sleep(0.5)

    print("[3/4] left=-0.5  -> bot SHOULD turn RIGHT (right wheel slows)")
    d.update(forward=0.3, left=-0.5)
    time.sleep(2)
    d.stop()
    time.sleep(0.5)

    print("[4/4] full stop")
    d.stop()


if __name__ == "__main__":
    main()
