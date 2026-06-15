#!/bin/bash
# Entrypoint for the JetBot racing image.
# Each model folder under /workspace/models/<name>/ is FULLY self-contained
# (its own bot_driving.py, PUTDriver.py, config.yml, preprocess.py, postprocess.py).
# We just cd into the right folder and exec its bot_driving.py.
#
# Usage: docker run ... jetbot-racing:phase3 <command>
#   <command> ∈ { pilotnet_fl_wsampler | pilotnet_fl_wloss | pilotnet_fl_shift1
#                | pilotnet_motor_wsampler | dualhead_class_reg
#                | mobilenetv3_fl_pretrained | calibrate | setup | bash }
set -e

MODEL=${1:-pilotnet_fl_wsampler}

case "$MODEL" in
  pilotnet_fl_wsampler|pilotnet_fl_wloss|pilotnet_fl_shift1|\
  pilotnet_motor_wsampler|dualhead_class_reg|mobilenetv3_fl_pretrained)
    cd "/workspace/models/$MODEL"
    exec python3 bot_driving.py
    ;;
  calibrate)
    exec python3 /workspace/calibrate.py --config /workspace/config.yml
    ;;
  setup)
    exec python3 /workspace/setup_differential.py --output /workspace/config.yml
    ;;
  bash|sh)
    exec bash
    ;;
  *)
    cat <<EOF
[run.sh] Unknown command: $MODEL

Models:
  pilotnet_fl_wsampler        (recommended baseline)
  pilotnet_fl_wloss
  pilotnet_fl_shift1
  pilotnet_motor_wsampler
  dualhead_class_reg
  mobilenetv3_fl_pretrained

Special commands:
  setup       Quick differential setup (JetBot 1-4 presets)
  calibrate   Interactive keyboard calibration of differential
  bash        Drop into a shell for debugging
EOF
    exit 1
    ;;
esac
