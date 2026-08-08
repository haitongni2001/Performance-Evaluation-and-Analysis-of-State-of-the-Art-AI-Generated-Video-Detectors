import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000_all_opensource.csv")

ROOT = Path(r"D:\HaitongNi\MEngProject\repos\DeMamba")
FRAME_ROOT = ROOT / "GenVideo" / "aigvd_common1000"
DATASET_DIR = ROOT / "GenVideo" / "datasets"

NUM_FRAMES = 8
SEED = 42

FRAME_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path, out_dir, num_frames=8):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= num_frames:
        return list(range(1, num_frames + 1))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    indices = np.linspace(0, total - 1, num_frames).astype(int)

    saved_ids = []
    last_frame = None

    for out_id, frame_idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            if last_frame is None:
                continue
            frame = last_frame
        else:
            last_frame = frame

        out_path = out_dir / f"{out_id}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_ids.append(out_id)

    cap.release()

    if len(saved_ids) == 0:
        raise RuntimeError(f"Could not decode any frames: {video_path}")

    while len(saved_ids) < num_frames:
        last_id = saved_ids[-1]
        new_id = len(saved_ids) + 1
        src = out_dir / f"{last_id}.jpg"
        dst = out_dir / f"{new_id}.jpg"
        dst.write_bytes(src.read_bytes())
        saved_ids.append(new_id)

    return list(range(1, num_frames + 1))


def main():
    df = pd.read_csv(INPUT_CSV)

    rows = []
    failed = []

    real_count = 0
    fake_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing DeMamba frames"):
        label_name = row["label_name"]
        label = int(row["label"])

        if label_name == "real":
            real_count += 1
            vid_name = f"real_{real_count:05d}"
            subdir = "real"
        else:
            fake_count += 1
            vid_name = f"fake_{fake_count:05d}"
            subdir = "fake"

        # content_path should be relative to GenVideo/
        # dataloader.py will look for: GenVideo/{content_path}/{frame_id}.jpg
        content_path = f"aigvd_common1000/{subdir}/{vid_name}"
        out_dir = FRAME_ROOT / subdir / vid_name

        try:
            frame_seq = extract_frames(row["path"], out_dir, NUM_FRAMES)

            rows.append({
                "content_path": content_path,
                "label": label,
                "frame_seq": str(frame_seq),
                "image_path": content_path,
                "video_id": row.get("video_id", ""),
                "label_name": row.get("label_name", ""),
                "source_type": row.get("source_type", ""),
                "task_type": row.get("task_type", ""),
                "generator_id": row.get("generator_id", ""),
                "original_path": row["path"],
            })

        except Exception as e:
            failed.append({
                "path": row["path"],
                "label": label,
                "error": str(e),
            })
            print("Failed:", row["path"], e)

    out = pd.DataFrame(rows)

    train_df, val_df = train_test_split(
        out,
        test_size=0.5,
        random_state=SEED,
        stratify=out["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(DATASET_DIR / "train.csv", index=False)
    val_df.to_csv(DATASET_DIR / "val_id.csv", index=False)

    # Also save explicit copies for clarity
    train_df.to_csv(DATASET_DIR / "aigvd_train.csv", index=False)
    val_df.to_csv(DATASET_DIR / "aigvd_val.csv", index=False)

    if failed:
        pd.DataFrame(failed).to_csv(DATASET_DIR / "aigvd_failed.csv", index=False)

    print("Saved:")
    print(DATASET_DIR / "train.csv")
    print(DATASET_DIR / "val_id.csv")
    print("Train:", len(train_df), train_df["label"].value_counts().to_dict())
    print("Val:", len(val_df), val_df["label"].value_counts().to_dict())
    print("Failed:", len(failed))


if __name__ == "__main__":
    main()