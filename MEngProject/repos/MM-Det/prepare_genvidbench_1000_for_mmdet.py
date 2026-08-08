import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

INPUT_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\metadata\genvidbench_common1000.csv")

OUT_ROOT = Path(r"D:\HaitongNi\MEngProject\repos\MM-Det\results\genvidbench_common1000\GenVidBench_common1000_raw")
DATASET_NAME = "genvidbench_common1000"

REAL_DIR = OUT_ROOT / DATASET_NAME / "0_real"
FAKE_DIR = OUT_ROOT / DATASET_NAME / "1_fake"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

real = df[df["label_name"] == "real"].copy()
fake = df[df["label_name"] == "fake"].copy()

assert len(real) == 500, len(real)
assert len(fake) == 500, len(fake)

for i, (_, row) in enumerate(tqdm(real.iterrows(), total=len(real), desc="Copying real"), start=1):
    dst = REAL_DIR / f"real_{i:05d}.mp4"
    if not dst.exists():
        shutil.copy2(row["path"], dst)

for i, (_, row) in enumerate(tqdm(fake.iterrows(), total=len(fake), desc="Copying fake"), start=1):
    dst = FAKE_DIR / f"fake_{i:05d}.mp4"
    if not dst.exists():
        shutil.copy2(row["path"], dst)

print("Saved:", OUT_ROOT)
print("Real:", len(list(REAL_DIR.glob("*.mp4"))))
print("Fake:", len(list(FAKE_DIR.glob("*.mp4"))))