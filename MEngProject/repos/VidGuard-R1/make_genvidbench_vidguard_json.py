import json
from pathlib import Path
import pandas as pd

INPUT_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\metadata\genvidbench_common1000.csv")
OUTPUT_JSON = Path(r"D:\HaitongNi\MEngProject\repos\VidGuard-R1\src\r1-v\Video-Ours-data\test_r1_genvidbench_common1000.json")

df = pd.read_csv(INPUT_CSV)

items = []

problem = (
    "Please determine whether the given video is AI-generated or real.\n"
)

options = [
    "A. The video is AI-generated/fake.",
    "B. The video is real."
]

for idx, row in df.iterrows():
    label = int(row["label"])

    # eval_bench.py maps A -> 1 and B -> 0
    answer = "A" if label == 1 else "B"

    items.append({
        "id": idx,
        "path": row["path"],
        "data_type": "video",
        "problem_type": "multiple choice",
        "problem": problem,
        "options": options,
        "solution": f"<answer>{answer}</answer>",
        "label": label,
        "label_name": row.get("label_name", ""),
        "generator_id": row.get("generator_id", ""),
        "pair": row.get("pair", ""),
        "rel_path": row.get("rel_path", ""),
    })

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)

print("Saved:", OUTPUT_JSON)
print("Rows:", len(items))
print("Label counts:")
print(df["label"].value_counts().sort_index().to_string())
print("Generator counts:")
print(df.groupby(["label_name", "generator_id"]).size().to_string())