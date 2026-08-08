import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"D:\HaitongNi\MEngProject_SecondBench\03_NSG-VD\dataset")
NUM_FRAMES = 8

VIDEO_REAL_DIR = ROOT / "video" / "real" / "MSR-VTT"
VIDEO_FAKE_DIR = ROOT / "video" / "fake" / "AIGV_OpenSource_All"

FRAME_REAL_TEST_DIR = ROOT / "video_frames" / "real" / "MSR-VTT" / "test"
FRAME_REAL_VAL_DIR = ROOT / "video_frames" / "real" / "MSR-VTT" / "val"
FRAME_FAKE_TEST_DIR = ROOT / "video_frames" / "fake" / "AIGV_OpenSource_All" / "test"


def extract_frames(video_path: Path, out_dir: Path, num_frames=8):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.*"))
    if len(existing) >= num_frames:
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open:", video_path)
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("No frames:", video_path)
        cap.release()
        return False

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

        out_path = out_dir / f"{i:05d}.png"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved > 0


def process_videos(video_dir: Path, frame_dir: Path, desc: str):
    videos = sorted(video_dir.glob("*.mp4"))
    ok = 0
    fail = 0

    for vp in tqdm(videos, desc=desc):
        out_dir = frame_dir / vp.stem
        if extract_frames(vp, out_dir, NUM_FRAMES):
            ok += 1
        else:
            fail += 1

    print(desc, "ok:", ok, "fail:", fail)
    return ok, fail


def main():
    FRAME_REAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_REAL_VAL_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_FAKE_TEST_DIR.mkdir(parents=True, exist_ok=True)

    process_videos(VIDEO_REAL_DIR, FRAME_REAL_TEST_DIR, "real test frames")
    process_videos(VIDEO_FAKE_DIR, FRAME_FAKE_TEST_DIR, "fake test frames")

    # NSG-VD also needs real val as reference.
    # Use the same real videos for val/reference, matching the AIGVDBench adaptation setting.
    real_test_dirs = sorted([p for p in FRAME_REAL_TEST_DIR.iterdir() if p.is_dir()])
    for src in tqdm(real_test_dirs, desc="copy real test frames to real val"):
        dst = FRAME_REAL_VAL_DIR / src.name
        if dst.exists():
            continue
        shutil.copytree(src, dst)

    print("Done.")
    print("real test frame dirs:", len(list(FRAME_REAL_TEST_DIR.iterdir())))
    print("real val frame dirs:", len(list(FRAME_REAL_VAL_DIR.iterdir())))
    print("fake test frame dirs:", len(list(FRAME_FAKE_TEST_DIR.iterdir())))


if __name__ == "__main__":
    main()