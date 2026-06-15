import pandas as pd
import os

root_dir = 'put_jetbot_dataset/dataset/train'  # your train dir

val2s = []
for folder_name in os.listdir(root_dir):
    csv_path = os.path.join(root_dir, f"{folder_name}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val2s.extend(df.iloc[:, 1].tolist())

import collections
buckets = collections.Counter()
for v in val2s:
    b = round(round(v * 10) / 10, 1)  # round to nearest 0.1
    buckets[b] += 1

for k in sorted(buckets):
    bar = '█' * (buckets[k] // 20)
    print(f"  {k:+.1f} | {bar} {buckets[k]}")