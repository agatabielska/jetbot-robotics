# jetbot-robotics

Imitation-learning pipeline for the JetBot racing task. A model is trained to predict `(forward, left)` control signals from camera frames and deployed on the robot via ONNX Runtime.

## Setup

```bash
uv sync
```

## Directory structure

```
dataset_annotated_final/   # training data (images + CSV labels) — not in repo, see below
dataset_annotated_initial/ # first annotated version — not in repo, see below
put_jetbot_dataset/        # raw course dataset — not in repo, see below

models/
  output/                  # checkpoints and ONNX files written here by default
  default-data/             # models trained on basic data
  initial-own-data/        # models trained on initial own annotated dataset
  final-own-data/          # models trained on final own annotated dataset

src/                       # training and on-robot scripts

experiments/               # all not that succesfull approaches, do not enter if you are pedantic
```

## Datasets

Dataset directories are excluded from the repository (`.gitignore`). There are three sources of data:

**Course dataset** (`put_jetbot_dataset/`) - the default dataset provided for the course.

**Annotated datasets** - versions of the data annotated by us using [data-annotation.ipynb](data-annotation.ipynb). 

Available [here](https://drive.google.com/drive/folders/1GUi_b3LJ-y4BEhcQb2xYh7fnYyso9Zgp?usp=sharing).

- `dataset_annotated_initial/` - first annotation pass.
- `dataset_annotated_final/` - annotation created for the final class session.

To use a specific dataset, pass `--dataset` to the training script.

## Training

Run from the `src/` directory:

```bash
cd src

# Default: shufflenet, 96×96, 50 epochs, dataset at ../dataset_annotated_final
uv run train.py

# Override any argument
uv run train.py --model mobilenet --epochs 40 --lr 3e-4
uv run train.py --model tiny --epochs 30
uv run train.py --dataset /path/to/custom/dataset --out-dir /path/to/output
```

The best checkpoint is saved to `../models/output/best_<model>.pt` and an ONNX file is exported automatically at the end.



## On-robot deployment

In jetbot environment run the driving script from `src/`:

```bash
python3 bot_driving.py
```

Robot parameters (`max_speed`, `max_steering`, latency compensation, etc.) are configured in the corresponding `.yml` file.