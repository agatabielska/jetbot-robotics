"""Session-level train/val split.

Random per-frame splits leak (consecutive frames are near-identical), so
we hold out whole sessions. 80/20 over sessions with a seeded shuffle.

Iteration 2 changes:
- Filter dropped frames with `throttle <= STATIONARY_THROTTLE` (was: only reverse).
  This removes both stationary and reverse frames so the model never learns
  "default = don't move". ~75% of frames are dropped.
- Val MUST contain one of LEFT_TURN_SESSIONS so val measures left-turn
  generalization, not just right-turn-loop memorization.
"""

import glob
import os
import random

import pandas as pd


STATIONARY_THROTTLE = 0.05
LEFT_TURN_SESSIONS = (
    "1652875851.3497071",
    "1652876206.2541456",
    "1653043202.5073502",
)


def _load_session(csv_path: str, img_dir: str):
    df = pd.read_csv(csv_path, header=None, names=["frame", "steer", "throttle"])
    samples = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, f"{int(row['frame']):04d}.jpg")
        if not os.path.exists(img_path):
            continue
        samples.append((img_path, float(row["steer"]), float(row["throttle"])))
    return samples


def build_splits(
    data_dir: str,
    val_frac: float = 0.2,
    seed: int = 42,
    filter_reverse: bool = True,
):
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"no CSVs in {data_dir}")

    session_samples = {}
    for csv_path in csvs:
        sess = os.path.basename(csv_path).replace(".csv", "")
        img_dir = os.path.join(data_dir, sess)
        if not os.path.isdir(img_dir):
            print(f"  WARN: no image folder for {sess}, skipping")
            continue
        session_samples[sess] = _load_session(csv_path, img_dir)

    sessions = sorted(session_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(sessions)

    n_val = max(1, int(round(len(sessions) * val_frac)))
    initial_val = sessions[:n_val]
    initial_train = sessions[n_val:]

    forced = [s for s in LEFT_TURN_SESSIONS if s in session_samples]
    if not forced:
        raise RuntimeError(
            "no LEFT_TURN_SESSIONS found in dataset — split.py is misconfigured"
        )

    if not any(s in initial_val for s in forced):
        chosen = rng.choice(forced)
        bumped = initial_val.pop()
        initial_val.append(chosen)
        if chosen in initial_train:
            initial_train.remove(chosen)
        initial_train.append(bumped)
        print(
            f"  forced-val: injected left-turn session {chosen}, "
            f"bumped {bumped} back to train"
        )

    val_sess = set(initial_val)
    train_sess = set(initial_train)
    assert not (val_sess & train_sess), "session overlap between splits"

    train_samples, val_samples = [], []
    pre_total = 0
    dropped = 0
    for sess, samples in session_samples.items():
        pre_total += len(samples)
        kept = []
        for s in samples:
            if filter_reverse and s[2] <= STATIONARY_THROTTLE:
                dropped += 1
                continue
            kept.append(s)
        if sess in train_sess:
            train_samples.extend(kept)
        else:
            val_samples.extend(kept)

    print(f"sessions: {len(sessions)} total | {len(train_sess)} train | {len(val_sess)} val")
    print(f"  val sessions: {sorted(val_sess)}")
    if filter_reverse:
        print(
            f"  filter throttle <= {STATIONARY_THROTTLE}: "
            f"kept {pre_total - dropped} / {pre_total} (dropped {dropped})"
        )
    print(f"frames: {len(train_samples)} train | {len(val_samples)} val")
    return train_samples, val_samples


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filter_reverse", action="store_true", default=True)
    args = ap.parse_args()
    build_splits(args.data_dir, seed=args.seed, filter_reverse=args.filter_reverse)
