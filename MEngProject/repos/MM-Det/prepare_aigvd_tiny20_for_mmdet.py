import shutil
from pathlib import Path
import pandas as pd

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000_all_opensource.csv")
OUT_ROOT = Path(r"D:\HaitongNi\MEngProject\mmdet_datasets\AIGVDBench_tiny20_raw")
DATASET_NAME = "aigvd_tiny20"

REAL_DIR = OUT_ROOT / DATASET_NAME / "0_real"
FAKE_DIR = OUT_ROOT / DATASET_NAME / "1_fake"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

real = df[df["label_name"] == "real"].head(10)
fake = df[df["label_name"] == "fake"].head(10)

for i, (_, row) in enumerate(real.iterrows(), start=1):
    dst = REAL_DIR / f"real_{i:05d}.mp4"
    if not dst.exists():
        shutil.copy2(row["path"], dst)

for i, (_, row) in enumerate(fake.iterrows(), start=1):
    dst = FAKE_DIR / f"fake_{i:05d}.mp4"
    if not dst.exists():
        shutil.copy2(row["path"], dst)

print("Saved:", OUT_ROOT)
print("Real:", len(list(REAL_DIR.glob('*.mp4'))))
print("Fake:", len(list(FAKE_DIR.glob('*.mp4'))))