import re
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(r"D:\HaitongNi\MEngProject\repos\MM-Det\results\genvidbench_common1000\GenVidBench_common1000_recons_cap64\genvidbench_common1000")
MIN_FRAMES = 10
LABELS = ["0_real", "1_fake"]

pattern = re.compile(r"^(.*)_(\d+)\.jpg$", re.IGNORECASE)


def group_frames(folder: Path):
    groups = defaultdict(dict)
    for p in folder.glob("*.jpg"):
        m = pattern.match(p.name)
        if not m:
            continue
        prefix = m.group(1)
        idx = int(m.group(2))
        groups[prefix][idx] = p
    return groups


def pad_label(label):
    orig_dir = ROOT / label / "original"
    rec_dir = ROOT / label / "recons"

    orig_groups = group_frames(orig_dir)
    rec_groups = group_frames(rec_dir)

    prefixes = sorted(set(orig_groups.keys()) & set(rec_groups.keys()))

    padded = 0
    already_ok = 0
    missing_pair = 0

    for prefix in tqdm(prefixes, desc=f"padding {label}"):
        orig_idxs = sorted(orig_groups[prefix].keys())
        rec_idxs = sorted(rec_groups[prefix].keys())
        common_idxs = sorted(set(orig_idxs) & set(rec_idxs))

        if not common_idxs:
            missing_pair += 1
            continue

        current_n = max(common_idxs)

        if current_n >= MIN_FRAMES:
            already_ok += 1
            continue

        last_idx = common_idxs[-1]
        last_orig = orig_groups[prefix][last_idx]
        last_rec = rec_groups[prefix][last_idx]

        for new_idx in range(current_n + 1, MIN_FRAMES + 1):
            shutil.copy2(last_orig, orig_dir / f"{prefix}_{new_idx}.jpg")
            shutil.copy2(last_rec, rec_dir / f"{prefix}_{new_idx}.jpg")

        padded += 1

    print(label)
    print("already_ok:", already_ok)
    print("padded:", padded)
    print("missing_pair:", missing_pair)


def main():
    for label in LABELS:
        pad_label(label)

    print("Done. All videos should now have at least 10 frames.")


if __name__ == "__main__":
    main()