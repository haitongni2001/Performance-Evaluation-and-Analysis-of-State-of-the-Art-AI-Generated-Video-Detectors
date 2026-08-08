import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_CSV = Path(r"D:\HaitongNi\MEngProject\repos\DeMamba\results\genvidbench_common1000\GenVidBench_XCLIP_DeMamba_small\Epoch_19_predictions.csv")
META_CSV = Path(r"D:\HaitongNi\MEngProject\repos\DeMamba\GenVideo\datasets\genvidbench_val.csv")

OUT_CSV = Path(r"D:\HaitongNi\MEngProject\repos\DeMamba\results\genvidbench_common1000\GenVidBench_XCLIP_DeMamba_small\demamba_genvidbench_common1000_epoch19_generator_breakdown.csv")

pred = pd.read_csv(PRED_CSV)
meta = pd.read_csv(META_CSV)

print("Prediction columns:", pred.columns.tolist())
print("Metadata columns:", meta.columns.tolist())

# Prediction CSV has:
# data_path, actual_label, predicted_label, pred_prob_fake
# Metadata CSV has:
# content_path, label, generator_id, etc.
df = pd.concat([pred.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)

df["y_true"] = df["actual_label"].astype(int)
df["score_fake"] = df["pred_prob_fake"].astype(float)
df["pred_label"] = df["predicted_label"].astype(int)

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

print("\nSaved:", OUT_CSV)
print()
print(out.to_string(index=False))

print("\nCheck:")
print("pred rows:", len(pred))
print("meta rows:", len(meta))
print("merged rows:", len(df))
print("real:", len(real))
print("fake:", len(fake))
print("missing generator_id:", df["generator_id"].isna().sum())