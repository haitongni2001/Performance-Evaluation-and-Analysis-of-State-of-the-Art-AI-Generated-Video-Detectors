import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_CSV = Path(r"D:\HaitongNi\MEngProject\repos\NSG-VD\results\genvidbench_common1000\default\nsgvd_genvidbench_common1000_predictions.csv")
OUT_CSV = Path(r"D:\HaitongNi\MEngProject\repos\NSG-VD\results\genvidbench_common1000\default\nsgvd_genvidbench_common1000_generator_breakdown.csv")

df = pd.read_csv(PRED_CSV)

# In NSG-VD, larger raw_score indicates more fake-like. Default threshold is raw_score > 1.
df["score_fake"] = df["raw_score"].astype(float)
df["y_true"] = df["label"].astype(int)
df["pred_label"] = (df["score_fake"] > 1).astype(int)

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
        "fake_recall_at_1": (g["pred_label"] == 1).mean(),
        "mean_score_fake": g["score_fake"].mean(),
        "auc_vs_all_real": auc,
        "ap_vs_all_real": ap,
    })

out = pd.DataFrame(rows).sort_values("fake_recall_at_1")
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("Saved:", OUT_CSV)
print()
print(out.to_string(index=False))

print("\nCheck:")
print("rows:", len(df))
print("real:", len(real))
print("fake:", len(fake))
print("missing generator_id:", df["generator_id"].isna().sum())