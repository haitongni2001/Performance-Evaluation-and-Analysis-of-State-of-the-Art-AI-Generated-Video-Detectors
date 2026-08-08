import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_6000.csv")

OUT_ROOT = Path(r"D:\HaitongNi\MEngProject\nsgvd_datasets\AIGVDBench_open_source_6000")

REAL_MODEL = "MSR-VTT"
FAKE_MODEL = "AIGV_OpenSource_All"

VIDEO_REAL_DIR = OUT_ROOT / "video" / "real" / REAL_MODEL
VIDEO_FAKE_DIR = OUT_ROOT / "video" / "fake" / FAKE_MODEL

SPLIT_REAL_DIR = OUT_ROOT / "split" / "real" / REAL_MODEL
SPLIT_FAKE_DIR = OUT_ROOT / "split" / "fake" / FAKE_MODEL

for p in [VIDEO_REAL_DIR, VIDEO_FAKE_DIR, SPLIT_REAL_DIR, SPLIT_FAKE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def copy_video(src_path, dst_dir, prefix, idx):
    src_path = Path(src_path)
    new_name = f"{prefix}_{idx:05d}.mp4"
    dst_path = dst_dir / new_name

    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)

    return new_name


def main():
    df = pd.read_csv(INPUT_CSV)

    real_df = df[df["label_name"] == "real"].copy()
    fake_df = df[df["label_name"] == "fake"].copy()

    print("Real:", len(real_df))
    print("Fake:", len(fake_df))
    print("Fake generator counts:")
    print(fake_df["generator_id"].value_counts())

    assert len(real_df) == 3000, f"Expected 3000 real, got {len(real_df)}"
    assert len(fake_df) == 3000, f"Expected 3000 fake, got {len(fake_df)}"

    real_ids = []
    fake_ids = []

    for i, (_, row) in enumerate(tqdm(real_df.iterrows(), total=len(real_df), desc="Copying real"), start=1):
        new_name = copy_video(row["path"], VIDEO_REAL_DIR, "real", i)
        real_ids.append(new_name)

    for i, (_, row) in enumerate(tqdm(fake_df.iterrows(), total=len(fake_df), desc="Copying fake"), start=1):
        new_name = copy_video(row["path"], VIDEO_FAKE_DIR, "fake", i)
        fake_ids.append(new_name)

    # NSG-VD needs test_ids.txt and val_ids.txt for real reference.
    for mode in ["test", "val"]:
        with open(SPLIT_REAL_DIR / f"{mode}_ids.txt", "w", encoding="utf-8") as f:
            for x in real_ids:
                f.write(x + "\n")

    with open(SPLIT_FAKE_DIR / "test_ids.txt", "w", encoding="utf-8") as f:
        for x in fake_ids:
            f.write(x + "\n")

    print("Saved NSG-VD dataset:")
    print(OUT_ROOT)
    print("Real videos:", len(real_ids))
    print("Fake videos:", len(fake_ids))


if __name__ == "__main__":
    main()