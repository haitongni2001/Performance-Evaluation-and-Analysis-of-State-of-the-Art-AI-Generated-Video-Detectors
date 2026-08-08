import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

REAL_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\02_D3\metadata\genvidbench_common1000_real.csv")
FAKE_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\02_D3\metadata\genvidbench_common1000_fake.csv")

OUT_ROOT = Path(r"D:\HaitongNi\MEngProject_SecondBench\02_D3\cache\frames")
NUM_FRAMES = 16


def safe_name(s):
    s = str(s).replace("\\", "/")
    s = s.replace(":", "_").replace("/", "_").replace(" ", "_")
    s = s.replace("___", "__")
    return s[:180]


def extract_frames(video_path, out_dir, num_frames=16):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.jpg"))
    if len(existing) >= num_frames:
        return str(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    indices = np.linspace(0, total - 1, num_frames).astype(int)

    saved = 0
    last_frame = None

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            if last_frame is None:
                continue
            frame = last_frame
        else:
            last_frame = frame

        out_path = out_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()

    if saved == 0:
        raise RuntimeError(f"Failed to save frames: {video_path}")

    return str(out_dir)


def process_csv(csv_path, label_name):
    df = pd.read_csv(csv_path)
    new_rows = []
    failed = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {label_name} frames"):
        video_path = Path(row["path"])

        sample_id = safe_name(row.get("rel_path", video_path.stem))
        out_dir = OUT_ROOT / label_name / sample_id

        try:
            frame_dir = extract_frames(video_path, out_dir, NUM_FRAMES)
            row["content_path"] = frame_dir
            row["target"] = int(row["label"])
            new_rows.append(row)
        except Exception as e:
            failed.append({
                "path": str(video_path),
                "error": str(e),
            })
            print("Failed:", video_path, e)

    out_df = pd.DataFrame(new_rows)
    out_csv = csv_path.with_name(csv_path.stem + "_frames.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if failed:
        fail_csv = csv_path.with_name(csv_path.stem + "_failed.csv")
        pd.DataFrame(failed).to_csv(fail_csv, index=False, encoding="utf-8-sig")
        print("Failed saved:", fail_csv)

    print("Saved:", out_csv)
    print("Rows:", len(out_df))
    return out_csv


if __name__ == "__main__":
    real_out = process_csv(REAL_CSV, "real")
    fake_out = process_csv(FAKE_CSV, "fake")

    print("\nUse these CSVs for D3:")
    print(real_out)
    print(fake_out)