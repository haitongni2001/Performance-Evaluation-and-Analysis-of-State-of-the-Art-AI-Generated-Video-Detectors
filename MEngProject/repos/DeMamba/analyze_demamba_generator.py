import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_CSV = Path(r"results\AIGVD_XCLIP_DeMamba_small\Epoch_19_predictions.csv")
META_CSV = Path(r"GenVideo\datasets\val_id.csv")

OUT_CSV = Path(r"D:\HaitongNi\MEngProject\results\demamba_aigvd_common1000_generator_breakdown.csv")

pred = pd.read_csv(PRED_CSV)
meta = pd.read_csv(META_CSV)

def norm_path(x):
    x = str(x).replace("\\", "/")
    # remove possible frame suffix if any
    if "__" in x:
        x = x.split("__")[0]
    return x

pred["content_path_norm"] = pred["data_path"].apply(norm_path)
meta["content_path_norm"] = meta["content_path"].apply(norm_path)

df = pred.merge(meta, on="content_path_norm", how="left", suffixes=("", "_meta"))

# Use prediction CSV labels as source of truth for evaluation
df["y_true"] = df["actual_label"].astype(int)
df["score_fake"] = df["pred_prob_fake"].astype(float)
df["pred_label"] = (df["score_fake"] >= 0.5).astype(int)
df["correct"] = df["pred_label"] == df["y_true"]

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
        "fake_recall_at_0.5": (g["pred_label"] == 1).mean(),
        "mean_score_fake": g["score_fake"].mean(),
        "auc_vs_all_real": auc,
        "ap_vs_all_real": ap,
    })

out = pd.DataFrame(rows).sort_values("fake_recall_at_0.5")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("Saved:", OUT_CSV)
print()
print(out.to_string(index=False))

print("\nMerge check:")
print("pred rows:", len(pred))
print("merged rows:", len(df))
print("missing generator_id:", df["generator_id"].isna().sum())