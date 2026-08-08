import json
import pandas as pd
from pathlib import Path

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000_all_opensource.csv")
OUTPUT_JSON = Path(r"D:\HaitongNi\MEngProject\vidguard_data\aigvd_all_opensource_1000.json")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

samples = []

for _, row in df.iterrows():
    label_name = row["label_name"]

    # A = AI-generated / fake, B = Real
    if label_name == "fake":
        solution = "<answer>A</answer>"
    elif label_name == "real":
        solution = "<answer>B</answer>"
    else:
        raise ValueError(label_name)

    item = {
        "path": str(row["path"]).replace("\\", "/"),
        "data_type": "video",
        "problem_type": "multiple choice",
        "problem": "Is this video AI-generated or real?",
        "options": [
            "A. AI-generated video",
            "B. Real video"
        ],
        "solution": solution,

        # extra metadata for our own analysis
        "video_id": row["video_id"],
        "label": int(row["label"]),
        "label_name": row["label_name"],
        "source_type": row["source_type"],
        "task_type": row["task_type"],
        "generator_id": row["generator_id"],
        "split": row["split"],
    }

    samples.append(item)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print("Saved:", OUTPUT_JSON)
print("Total:", len(samples))
print(df["label_name"].value_counts())
print(df["generator_id"].value_counts())