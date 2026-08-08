import json
from pathlib import Path
import pandas as pd

VIDEO_ROOT = Path(r"F:\MEngDatasets\AIGVDBench\videos")
SPLIT_ROOT = Path(r"F:\MEngDatasets\AIGVDBench\metadata\Split")
OUTPUT_DIR = Path(r"F:\MEngDatasets\AIGVDBench\metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 读取官方 split 文件
video_to_split = {}

split_files = {
    "train": SPLIT_ROOT / "train.jsonl",
    "val": SPLIT_ROOT / "val.jsonl",
    "test": SPLIT_ROOT / "test.jsonl",
}

for split_name, split_path in split_files.items():
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            video_id = item["Video_id"]
            video_to_split[video_id] = split_name

print(f"Loaded official split records: {len(video_to_split)}")

# 2. 解析每个视频的来源信息
def parse_aigvd_path(path: Path):
    """
    根据路径判断 label/source_type/task_type/generator/generator_id.

    期望路径大概长这样：
    F:\\MEngDatasets\\AIGVDBench\\videos\\Real\\Real\\xxx.mp4
    F:\\MEngDatasets\\AIGVDBench\\videos\\ClosedSource\\Sora\\xxx.mp4
    F:\\MEngDatasets\\AIGVDBench\\videos\\OpenSource\\T2V\\Open-Sora\\xxx.mp4
    F:\\MEngDatasets\\AIGVDBench\\videos\\OpenSource\\I2V\\LTX\\xxx.mp4
    F:\\MEngDatasets\\AIGVDBench\\videos\\OpenSource\\V2V\\Cogvideox1.5\\xxx.mp4
    """

    parts = list(path.parts)

    # 默认值
    label = -1
    label_name = "unknown"
    source_type = "Unknown"
    task_type = "Unknown"
    generator = "Unknown"
    generator_id = "Unknown"

    if "Real" in parts:
        label = 0
        label_name = "real"
        source_type = "Real"
        task_type = "None"
        generator = "Real"
        generator_id = "Real"

    elif "ClosedSource" in parts:
        label = 1
        label_name = "fake"
        source_type = "ClosedSource"
        task_type = "None"

        idx = parts.index("ClosedSource")
        if idx + 1 < len(parts):
            generator = parts[idx + 1]

        generator_id = f"{source_type}_{generator}"

    elif "OpenSource" in parts:
        label = 1
        label_name = "fake"
        source_type = "OpenSource"

        idx = parts.index("OpenSource")

        if idx + 1 < len(parts):
            task_type = parts[idx + 1]  # T2V / I2V / V2V

        if idx + 2 < len(parts):
            generator = parts[idx + 2]

        generator_id = f"{source_type}_{task_type}_{generator}"

    return {
        "label": label,
        "label_name": label_name,
        "source_type": source_type,
        "task_type": task_type,
        "generator": generator,
        "generator_id": generator_id,
    }


# 3. 扫描所有本地 mp4
rows = []

for path in VIDEO_ROOT.rglob("*.mp4"):
    video_id = path.name
    info = parse_aigvd_path(path)
    split = video_to_split.get(video_id, "not_in_official_split")

    rows.append({
        "video_id": video_id,
        "path": str(path),
        "split": split,
        **info,
    })

df = pd.DataFrame(rows)

# 4. 保存全量索引
all_csv = OUTPUT_DIR / "aigvd_all_index.csv"
df.to_csv(all_csv, index=False, encoding="utf-8-sig")

print(f"Total local mp4 files: {len(df)}")
print(f"Saved all index to: {all_csv}")

# 5. 保存官方 test split 里的视频
test_df = df[df["split"] == "test"].copy()
test_csv = OUTPUT_DIR / "aigvd_test_index.csv"
test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")

print(f"Official test videos found locally: {len(test_df)}")
print(f"Saved test index to: {test_csv}")

# 6. 打印统计
print("\nSplit counts:")
print(df["split"].value_counts())

print("\nTest label counts:")
print(test_df["label_name"].value_counts())

print("\nTest source_type counts:")
print(test_df["source_type"].value_counts())

print("\nTest task_type counts:")
print(test_df["task_type"].value_counts())

print("\nTest generator_id counts:")
print(test_df["generator_id"].value_counts().head(50))