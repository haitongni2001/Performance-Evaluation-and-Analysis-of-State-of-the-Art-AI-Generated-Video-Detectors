import re
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

SRC_ROOT = Path(r"D:\HaitongNi\MEngProject\repos\MM-Det\results\genvidbench_common1000\GenVidBench_common1000_recons\genvidbench_common1000")
DST_ROOT = Path(r"D:\HaitongNi\MEngProject\repos\MM-Det\results\genvidbench_common1000\GenVidBench_common1000_recons_cap64\genvidbench_common1000")

MAX_FRAMES = 64
LABELS = ["0_real", "1_fake"]

pattern = re.compile(r"^(.*)_(\d+)\.jpg$", re.IGNORECASE)


def group_frames(folder: Path):
    groups = {}
    for p in folder.glob("*.jpg"):
        m = pattern.match(p.name)
        if not m:
            continue
        prefix = m.group(1)
        idx = int(m.group(2))
        groups.setdefault(prefix, {})[idx] = p
    return groups


def process_label(label):
    src_label = SRC_ROOT / label
    dst_label = DST_ROOT / label

    src_original = src_label / "original"
    src_recons = src_label / "recons"

    dst_original = dst_label / "original"
    dst_recons = dst_label / "recons"
    dst_original.mkdir(parents=True, exist_ok=True)
    dst_recons.mkdir(parents=True, exist_ok=True)

    orig_groups = group_frames(src_original)
    recons_groups = group_frames(src_recons)

    common_prefixes = sorted(set(orig_groups.keys()) & set(recons_groups.keys()))

    print(f"\n{label}: videos = {len(common_prefixes)}")

    total_copied = 0

    for prefix in tqdm(common_prefixes, desc=f"capping {label}"):
        orig_idxs = set(orig_groups[prefix].keys())
        recons_idxs = set(recons_groups[prefix].keys())
        common_idxs = sorted(orig_idxs & recons_idxs)

        if len(common_idxs) == 0:
            continue

        if len(common_idxs) > MAX_FRAMES:
            pick_pos = np.linspace(0, len(common_idxs) - 1, MAX_FRAMES).astype(int)
            selected = [common_idxs[i] for i in pick_pos]
        else:
            selected = common_idxs

        # Rename to contiguous indices so MM-Det does not look for missing sparse frame IDs.
        for new_i, old_i in enumerate(selected):
            new_name = f"{prefix}_{new_i + 1}.jpg"
            shutil.copy2(orig_groups[prefix][old_i], dst_original / new_name)
            shutil.copy2(recons_groups[prefix][old_i], dst_recons / new_name)
            total_copied += 1

    print(f"{label}: copied frame pairs = {total_copied}")


def main():
    for label in LABELS:
        process_label(label)

    print("\nSaved capped recons dataset:")
    print(DST_ROOT)


if __name__ == "__main__":
    main()