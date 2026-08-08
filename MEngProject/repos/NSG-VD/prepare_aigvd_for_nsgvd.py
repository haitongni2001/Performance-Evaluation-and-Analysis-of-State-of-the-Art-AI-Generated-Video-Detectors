import shutil
from pathlib import Path
import pandas as pd

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_6000.csv")

OUT_ROOT = Path(r"D:\HaitongNi\MEngProject\nsgvd_datasets\AIGVDBench_pilot100")

# NSG-VD expects this GenVideo-like structure
REAL_MODEL = "MSR-VTT"
FAKE_MODEL = "Sora"

N_REAL = 50
N_FAKE = 50

VIDEO_REAL_DIR = OUT_ROOT / "video" / "real" / REAL_MODEL
VIDEO_FAKE_DIR = OUT_ROOT / "video" / "fake" / FAKE_MODEL

SPLIT_REAL_DIR = OUT_ROOT / "split" / "real" / REAL_MODEL
SPLIT_FAKE_DIR = OUT_ROOT / "split" / "fake" / FAKE_MODEL

for p in [VIDEO_REAL_DIR, VIDEO_FAKE_DIR, SPLIT_REAL_DIR, SPLIT_FAKE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def copy_video(src_path, dst_dir, prefix):
    src_path = Path(src_path)
    stem = src_path.stem

    # Keep extension as .mp4
    new_name = f"{prefix}_{stem}.mp4"
    dst_path = dst_dir / new_name

    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)

    # split txt expects filename with .mp4 based on this repo
    return new_name


def main():
    df = pd.read_csv(INPUT_CSV)

    real_df = df[df["label_name"] == "real"].sample(n=N_REAL, random_state=42)

    # Pick one fake generator for pilot.
    # If OpenSource_T2V_Open-Sora exists, use it. Otherwise sample from all fake.
    fake_df_all = df[df["label_name"] == "fake"].copy()
    open_sora = fake_df_all[fake_df_all["generator_id"] == "OpenSource_T2V_Open-Sora"]

    if len(open_sora) >= N_FAKE:
        fake_df = open_sora.sample(n=N_FAKE, random_state=42)
    else:
        fake_df = fake_df_all.sample(n=N_FAKE, random_state=42)

    real_ids = []
    fake_ids = []

    for _, row in real_df.iterrows():
        new_name = copy_video(row["path"], VIDEO_REAL_DIR, "real")
        real_ids.append(new_name)

    for _, row in fake_df.iterrows():
        new_name = copy_video(row["path"], VIDEO_FAKE_DIR, "fake")
        fake_ids.append(new_name)

    # NSG-VD looks for mode_ids.txt, e.g., test_ids.txt / val_ids.txt
    # For reference real data, config may use val mode, so we write both val and test.
    for mode in ["test", "val"]:
        with open(SPLIT_REAL_DIR / f"{mode}_ids.txt", "w", encoding="utf-8") as f:
            for x in real_ids:
                f.write(x + "\n")

    with open(SPLIT_FAKE_DIR / "test_ids.txt", "w", encoding="utf-8") as f:
        for x in fake_ids:
            f.write(x + "\n")

    print("Saved NSG-VD pilot dataset:")
    print(OUT_ROOT)
    print("Real videos:", len(real_ids), VIDEO_REAL_DIR)
    print("Fake videos:", len(fake_ids), VIDEO_FAKE_DIR)
    print("Real split:", SPLIT_REAL_DIR)
    print("Fake split:", SPLIT_FAKE_DIR)


if __name__ == "__main__":
    main()