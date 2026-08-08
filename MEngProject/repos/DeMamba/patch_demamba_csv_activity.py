import pandas as pd
from pathlib import Path

files = [
    "GenVideo/datasets/train.csv",
    "GenVideo/datasets/val_id.csv",
    "GenVideo/datasets/val_ood.csv",
    "GenVideo/datasets/aigvd_train.csv",
    "GenVideo/datasets/aigvd_val.csv",
]

for f in files:
    p = Path(f)
    if not p.exists():
        print("missing:", f)
        continue

    df = pd.read_csv(p)

    if "activity_id" not in df.columns:
        df["activity_id"] = 0

    df.to_csv(p, index=False)
    print("patched:", f, df.shape)
    print(df.columns.tolist())