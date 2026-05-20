# JetBot Calibration — Quick Guide

JetBot's left/right wheels don't have encoders; manufacturing tolerance means one wheel
runs faster than the other at the same PWM. **Without calibration, the bot drifts** —
this can cost the race more than any model imperfection.

## When to calibrate
- First time you use a particular physical JetBot
- After replacing a wheel/motor
- When the bot consistently drifts to one side at constant `forward=0.5, left=0`

## Procedure (5 minutes)

### 1. Prep
- Flat surface, ~1m of clear space ahead
- Power-on JetBot
- SSH into JetBot

### 2. Run the calibration script INSIDE the container
```bash
docker run --rm -it --runtime nvidia \
    -v $HOME/config.yml:/workspace/config.yml \
    jetbot-racing:phase3 calibrate
```

### 3. Iterate
| Key | Action |
|---|---|
| `w` | Drive forward at constant `forward=0.5` (both wheels) |
| `s` | Stop |
| `q` / `e` | Decrease / increase `differential.left` by 0.05 |
| `a` / `d` | Decrease / increase `differential.right` by 0.05 |
| `r` | Reset to `(1.00, 1.00)` |
| `p` | Save current values to `config.yml` |
| `x` | Quit (auto-stops motors) |

### 4. Tuning logic
- Bot drifts **LEFT** at `w`?  → Left wheel is too slow OR right is too fast.
  - Try `e` (boost left). If still drifts, try `a` (slow right).
- Bot drifts **RIGHT**? Opposite — `q` (slow left) or `d` (boost right).

### 5. Sanity check
After saving, drive the bot ~1m with `w` again. If it stays within ~10 cm of center, calibration is good.

## Reference (from `PUT_jetbot/README.md`)
Course-author measurements for past JetBots:

| Vehicle | left | right |
|---|---|---|
| Jetbot 01 | 1.00 | 0.90 |
| Jetbot 02 | 0.85 | 1.00 |
| Jetbot 03 | 1.00 | 1.00 |
| Jetbot 04 | 1.00 | 1.00 |

If you know your JetBot number, you can pre-set these values directly in `config.yml`
and SKIP calibration — but verify with one `w` test before the race anyway.

## Troubleshooting
- **"Cannot import jetbot"** — you ran calibrate.py on a dev host, not on JetBot. Use Docker on real hardware.
- **"sshkeyboard not installed"** — included in our image; if missing run `pip3 install sshkeyboard`.
- **Robot doesn't move** — check `max_speed` in config.yml; min usable is ~0.18.
- **Wheels spin opposite to expectation** — wiring is reversed; report to the instructor.
