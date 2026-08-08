import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix

PRED_JSON = Path(r"D:\HaitongNi\MEngProject\repos\VidGuard-R1\results\genvidbench_common1000\vidguard_genvidbench_common1000_qwen25vl7b.json")
OUT_CSV = Path(r"D:\HaitongNi\MEngProject\repos\VidGuard-R1\results\genvidbench_common1000\vidguard_genvidbench_common1000_generator_breakdown.csv")


def extract_answer(text):
    if text is None:
        return ""
    text = str(text)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def normalize_pred_to_label(x):
    ans = extract_answer(x).strip().upper()

    # eval_bench.py uses A -> fake(1), B -> real(0)
    if ans == "A" or ans.startswith("A."):
        return 1
    if ans == "B" or ans.startswith("B."):
        return 0

    # fallback for natural language answers
    low = ans.lower()
    if "fake" in low or "ai-generated" in low or "ai generated" in low or "synthetic" in low:
        return 1
    if "real" in low or "authentic" in low:
        return 0

    return np.nan


def normalize_gt_to_label(sample):
    if "label" in sample:
        return int(sample["label"])

    sol = sample.get("solution", "")
    ans = extract_answer(sol).strip().upper()
    if ans == "A" or ans.startswith("A."):
        return 1
    if ans == "B" or ans.startswith("B."):
        return 0

    return np.nan


with open(PRED_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", data)

rows = []
for i, sample in enumerate(results):
    y_true = normalize_gt_to_label(sample)
    y_pred = normalize_pred_to_label(sample.get("prediction", sample.get("output", "")))

    rows.append({
        "idx": i,
        "path": sample.get("path", ""),
        "label": y_true,
        "label_name": "fake" if y_true == 1 else "real" if y_true == 0 else "unknown",
        "pred_label": y_pred,
        "generator_id": sample.get("generator_id", "unknown"),
        "pair": sample.get("pair", ""),
        "prediction_raw": sample.get("prediction", ""),
    })

df = pd.DataFrame(rows)

valid = df.dropna(subset=["label", "pred_label"]).copy()
valid["label"] = valid["label"].astype(int)
valid["pred_label"] = valid["pred_label"].astype(int)

real = valid[valid["label"] == 0].copy()
fake = valid[valid["label"] == 1].copy()

breakdown_rows = []

for gen, g in fake.groupby("generator_id"):
    pair = pd.concat([real, g], ignore_index=True)

    y_true = pair["label"].values
    y_pred = pair["pred_label"].values

    breakdown_rows.append({
        "generator_id": gen,
        "n_fake": len(g),
        "fake_recall": (g["pred_label"] == 1).mean(),
        "accuracy_vs_all_real": accuracy_score(y_true, y_pred),
        "precision_fake_vs_all_real": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake_vs_all_real": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake_vs_all_real": f1_score(y_true, y_pred, zero_division=0),
        "mcc_vs_all_real": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix_vs_all_real": confusion_matrix(y_true, y_pred).tolist(),
    })

out = pd.DataFrame(breakdown_rows).sort_values("fake_recall")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("Saved:", OUT_CSV)
print()
print(out.to_string(index=False))

print("\nOverall check:")
print("rows:", len(df))
print("valid:", len(valid))
print("invalid:", len(df) - len(valid))
print("real:", len(real))
print("fake:", len(fake))
print("missing generator_id:", valid["generator_id"].isna().sum())