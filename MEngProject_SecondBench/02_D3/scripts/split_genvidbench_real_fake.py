import pandas as pd
from pathlib import Path

IN = Path(r"D:\HaitongNi\MEngProject_SecondBench\metadata\genvidbench_common1000.csv")
OUT_DIR = Path(r"D:\HaitongNi\MEngProject_SecondBench\02_D3\metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

real = df[df["label"] == 0].copy()
fake = df[df["label"] == 1].copy()

real_out = OUT_DIR / "genvidbench_common1000_real.csv"
fake_out = OUT_DIR / "genvidbench_common1000_fake.csv"

real.to_csv(real_out, index=False, encoding="utf-8-sig")
fake.to_csv(fake_out, index=False, encoding="utf-8-sig")

print("Saved:", real_out, len(real))
print("Saved:", fake_out, len(fake))
print(real[["path", "label", "generator_id"]].head().to_string(index=False))
print(fake[["path", "label", "generator_id"]].head().to_string(index=False))