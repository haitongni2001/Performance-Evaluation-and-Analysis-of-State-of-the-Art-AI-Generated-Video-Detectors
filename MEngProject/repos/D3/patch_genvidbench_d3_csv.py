import pandas as pd
from pathlib import Path

files = [
    Path(r"D:\HaitongNi\MEngProject_SecondBench\02_D3\metadata\genvidbench_common1000_real.csv"),
    Path(r"D:\HaitongNi\MEngProject_SecondBench\02_D3\metadata\genvidbench_common1000_fake.csv"),
]

for p in files:
    df = pd.read_csv(p)

    # D3 dataloader expects content_path
    if "content_path" not in df.columns:
        df["content_path"] = df["path"]

    # Keep label column if needed
    if "target" not in df.columns:
        df["target"] = df["label"]

    df.to_csv(p, index=False, encoding="utf-8-sig")
    print("patched:", p)
    print(df.columns.tolist())
    print(df[["content_path", "label", "generator_id"]].head().to_string(index=False))