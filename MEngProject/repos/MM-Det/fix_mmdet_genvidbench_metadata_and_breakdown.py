import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_CSV = Path(r"D:\HaitongNi\MEngProject\repos\MM-Det\results\genvidbench_common1000\mmdet_genvidbench_common1000_cap64_predictions.csv")
META_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\metadata\genvidbench_common1000.csv")

OUT_PRED = Path(r"D:\HaitongNi\MEngProject\repos\MM-Det\results\genvidbench_common1000\mmdet_genvidbench_common1000_cap64_predictions_fixed_metadata.csv")
OUT_BREAKDOWN = Path(r"D:\HaitongNi\MEngProject\repos\MM-Det\results\genvidbench_common1000\mmdet_genvidbench_common1000_cap64_generator_breakdown_fixed.csv")

pred = pd.read_csv(PRED_CSV)
meta = pd.read_csv(META_CSV)

# MM-Det copied videos as real_00001 / fake_00001 etc.
# Reconstruct the same order from genvidbench_common1000.csv:
# real rows in CSV order -> real_00001...
# fake rows in CSV order -> fake_00001...
real_meta = meta[meta["label"] == 0].reset_index(drop=True).copy()
fake_meta = meta[meta["label"] == 1].reset_index(drop=True).copy()

real_meta["mmdet_id"] = [f"real_{i+1:05d}" for i in range(len(real_meta))]
fake_meta["mmdet_id"] = [f"fake_{i+1:05d}" for i in range(len(fake_meta))]

fixed_meta = pd.concat([real_meta, fake_meta], ignore_index=True)

# Drop wrong old metadata columns from the previous merge.
old_meta_cols = [
    "video_id", "path", "split", "label", "label_name",
    "source_type", "task_type", "generator", "generator_id"
]
for c in old_meta_cols:
    if c in pred.columns:
        pred = pred.drop(columns=[c])

fixed = pred.merge(
    fixed_meta,
    on="mmdet_id",
    how="left",
    suffixes=("", "_genvid")
)

if fixed["generator_id"].isna().sum() != 0:
    print("WARNING: missing generator_id:", fixed["generator_id"].isna().sum())

# Use MM-Det score_fake already produced by test_customized_savepred.py.
fixed["score_fake"] = fixed["score_fake"].astype(float)
fixed["y_true"] = fixed["y_true"].astype(int)

# Ensure pred_label exists.
if "pred_label" not in fixed.columns:
    fixed["pred_label"] = (fixed["score_fake"] >= 0.5).astype(int)
else:
    fixed["pred_label"] = fixed["pred_label"].astype(int)

fixed["correct"] = fixed["y_true"] == fixed["pred_label"]

fixed.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")

real = fixed[fixed["y_true"] == 0].copy()
fake = fixed[fixed["y_true"] == 1].copy()

rows = []
for gen, g in fake.groupby("generator_id"):
    pair = pd.concat([real, g], ignore_index=True)

    auc = np.nan
    ap = np.nan
    if pair["y_true"].nunique() == 2:
        auc = roc_auc_score(pair["y_true"], pair["score_fake"])
        ap = average_precision_score(pair["y_true"], pair["score_fake"])

    rows.append({
        "generator_id": gen,
        "n_fake": len(g),
        "fake_recall_at_0.5": (g["pred_label"] == 1).mean(),
        "mean_score_fake": g["score_fake"].mean(),
        "auc_vs_all_real": auc,
        "ap_vs_all_real": ap,
    })

out = pd.DataFrame(rows).sort_values("fake_recall_at_0.5")
out.to_csv(OUT_BREAKDOWN, index=False, encoding="utf-8-sig")

print("Saved fixed prediction CSV:")
print(OUT_PRED)

print("\nSaved fixed generator breakdown:")
print(OUT_BREAKDOWN)

print("\nGenerator breakdown:")
print(out.to_string(index=False))

print("\nCheck:")
print("pred rows:", len(pred))
print("meta rows:", len(meta))
print("fixed rows:", len(fixed))
print("real:", len(real))
print("fake:", len(fake))
print("missing generator_id:", fixed["generator_id"].isna().sum())
print("\nGenerator counts:")
print(fixed.groupby(["label_name", "generator_id"]).size().to_string())