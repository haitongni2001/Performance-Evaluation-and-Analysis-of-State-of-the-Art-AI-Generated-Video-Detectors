import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000_all_opensource.csv")

OUT_ROOT = Path(r"D:\HaitongNi\MEngProject\d3_datasets\AIGVDBench_all_opensource_1000")
FRAME_ROOT = OUT_ROOT / "frames"
CSV_ROOT = OUT_ROOT / "csv"

FRAME_ROOT.mkdir(parents=True, exist_ok=True)
CSV_ROOT.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 16


def safe_name(row):
    stem = Path(row["path"]).stem
    gen = str(row["generator_id"]).replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{gen}__{stem}"


def extract_frames(video_path, out_dir, num_frames=16):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= 8:
        return len(existing)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    indices = [int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]

    saved = 0
    for idx, frame_idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        out_path = out_dir / f"{idx}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def make_rows(df, label_name):
    sub = df[df["label_name"] == label_name].copy()
    rows = []

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"Processing {label_name}"):
        video_path = Path(row["path"])
        folder_name = safe_name(row)

        class_dir = "real" if label_name == "real" else "fake"
        frame_dir = FRAME_ROOT / class_dir / folder_name

        frame_count = extract_frames(video_path, frame_dir, NUM_FRAMES)

        if frame_count < 8:
            print(f"Skip, not enough frames: {video_path}")
            continue

        frame_seq = list(range(1, frame_count + 1))

        rows.append({
            "content_path": str(frame_dir),
            "image_path": str(frame_dir / "1.jpg"),
            "type_id": "Real Video" if label_name == "real" else "AI Video",
            "label": 0 if label_name == "real" else 1,
            "frame_len": frame_count,
            "frame_seq": str(frame_seq),
            "video_id": row["video_id"],
            "generator_id": row["generator_id"],
            "task_type": row["task_type"],
            "original_path": row["path"],
        })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT_CSV)

    print("Input label counts:")
    print(df["label_name"].value_counts())

    print("\nInput task type counts:")
    print(df["task_type"].value_counts())

    print("\nInput generator counts:")
    print(df["generator_id"].value_counts())

    real_csv = make_rows(df, "real")
    fake_csv = make_rows(df, "fake")

    real_out = CSV_ROOT / "real.csv"
    fake_out = CSV_ROOT / "fake.csv"

    real_csv.to_csv(real_out, index=False, encoding="utf-8-sig")
    fake_csv.to_csv(fake_out, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(real_out)
    print(fake_out)
    print("Real samples:", len(real_csv))
    print("Fake samples:", len(fake_csv))


if __name__ == "__main__":
    main()