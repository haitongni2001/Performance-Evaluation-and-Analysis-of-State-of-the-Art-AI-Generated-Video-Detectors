import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_CSV = Path(r"D:\HaitongNi\MEngProject\repos\D3\results\genvidbench_common1000\predictions_20260518_180250.csv")

REAL_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\02_D3\metadata\genvidbench_common1000_real_frames.csv")
FAKE_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\02_D3\metadata\genvidbench_common1000_fake_frames.csv")

OUT_CSV = Path(r"D:\HaitongNi\MEngProject\repos\D3\results\genvidbench_common1000\generator_breakdown_20260518_180250.csv")

pred = pd.read_csv(PRED_CSV)
meta = pd.concat([pd.read_csv(REAL_CSV), pd.read_csv(FAKE_CSV)], ignore_index=True)

print("Prediction columns:", pred.columns.tolist())
print("Metadata columns:", meta.columns.tolist())

# Try to identify score columns from D3 output.
# In our previous D3 script, fake_score and real_score are usually saved.
if "fake_score" in pred.columns:
    score_col = "fake_score"
elif "score_fake" in pred.columns:
    score_col = "score_fake"
elif "pred_score" in pred.columns:
    score_col = "pred_score"
else:
    # fallback: print columns and stop with a clear error
    raise RuntimeError("Could not find fake score column. Check prediction CSV columns above.")

# Identify path/content key.
if "content_path" in pred.columns:
    pred_key = "content_path"
elif "path" in pred.columns:
    pred_key = "path"
elif "video_path" in pred.columns:
    pred_key = "video_path"
else:
    # If prediction CSV has no path column, use row order.
    pred_key = None

if pred_key is not None and "content_path" in meta.columns:
    pred["join_key"] = pred[pred_key].astype(str).str.replace("\\", "/", regex=False)
    meta["join_key"] = meta["content_path"].astype(str).str.replace("\\", "/", regex=False)
    df = pred.merge(meta, on="join_key", how="left", suffixes=("", "_meta"))
else:
    # D3 eval order is real CSV followed by fake CSV, so row-order merge is acceptable
    # only if prediction CSV saved samples in dataloader order.
    df = pd.concat([pred.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)

# Label column
if "label" in df.columns:
    y_col = "label"
elif "actual_label" in df.columns:
    y_col = "actual_label"
elif "target" in df.columns:
    y_col = "target"
elif "true_label" in df.columns:
    y_col = "true_label"
else:
    raise RuntimeError("Could not find label column after merge.")

df["y_true"] = df[y_col].astype(int)
df["score_fake"] = df[score_col].astype(float)

# Use diagnostic best threshold from D3 result.
BEST_THRESHOLD = -4.303232
df["pred_label_best"] = (df["score_fake"] >= BEST_THRESHOLD).astype(int)

real = df[df["y_true"] == 0].copy()
fake = df[df["y_true"] == 1].copy()

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
        "fake_recall_at_best_threshold": (g["pred_label_best"] == 1).mean(),
        "mean_fake_score": g["score_fake"].mean(),
        "auc_vs_all_real": auc,
        "ap_vs_all_real": ap,
    })

out = pd.DataFrame(rows).sort_values("fake_recall_at_best_threshold")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("\nSaved:", OUT_CSV)
print()
print(out.to_string(index=False))

print("\nMerge check:")
print("pred rows:", len(pred))
print("metadata rows:", len(meta))
print("merged rows:", len(df))
print("missing generator_id:", df["generator_id"].isna().sum())