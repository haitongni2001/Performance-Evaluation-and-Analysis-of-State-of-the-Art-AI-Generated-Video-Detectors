import shutil
from pathlib import Path
import pandas as pd

INPUT_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\metadata\genvidbench_common1000.csv")
OUT_ROOT = Path(r"D:\HaitongNi\MEngProject_SecondBench\03_NSG-VD\dataset")

REAL_MODEL = "MSR-VTT"
FAKE_MODEL = "AIGV_OpenSource_All"

VIDEO_REAL_DIR = OUT_ROOT / "video" / "real" / REAL_MODEL
VIDEO_FAKE_DIR = OUT_ROOT / "video" / "fake" / FAKE_MODEL

SPLIT_REAL_DIR = OUT_ROOT / "split" / "real" / REAL_MODEL
SPLIT_FAKE_DIR = OUT_ROOT / "split" / "fake" / FAKE_MODEL

for p in [VIDEO_REAL_DIR, VIDEO_FAKE_DIR, SPLIT_REAL_DIR, SPLIT_FAKE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def copy_video(src_path, dst_dir, new_name):
    src_path = Path(src_path)
    dst_path = dst_dir / new_name

    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)

    return new_name


def main():
    df = pd.read_csv(INPUT_CSV)

    real_df = df[df["label"] == 0].reset_index(drop=True)
    fake_df = df[df["label"] == 1].reset_index(drop=True)

    assert len(real_df) == 500, f"Expected 500 real, got {len(real_df)}"
    assert len(fake_df) == 500, f"Expected 500 fake, got {len(fake_df)}"

    real_ids = []
    fake_ids = []

    for i, row in real_df.iterrows():
        new_name = f"real_{i+1:05d}.mp4"
        real_ids.append(copy_video(row["path"], VIDEO_REAL_DIR, new_name))

    for i, row in fake_df.iterrows():
        gen = str(row.get("generator_id", "fake")).replace("/", "_").replace("\\", "_")
        new_name = f"fake_{i+1:05d}_{gen}.mp4"
        fake_ids.append(copy_video(row["path"], VIDEO_FAKE_DIR, new_name))

    for mode in ["test", "val"]:
        with open(SPLIT_REAL_DIR / f"{mode}_ids.txt", "w", encoding="utf-8") as f:
            for x in real_ids:
                f.write(x + "\n")

    with open(SPLIT_FAKE_DIR / "test_ids.txt", "w", encoding="utf-8") as f:
        for x in fake_ids:
            f.write(x + "\n")

    print("Saved NSG-VD GenVidBench dataset:")
    print(OUT_ROOT)
    print("Real videos:", len(real_ids), VIDEO_REAL_DIR)
    print("Fake videos:", len(fake_ids), VIDEO_FAKE_DIR)
    print("Real split:", SPLIT_REAL_DIR)
    print("Fake split:", SPLIT_FAKE_DIR)


if __name__ == "__main__":
    main()