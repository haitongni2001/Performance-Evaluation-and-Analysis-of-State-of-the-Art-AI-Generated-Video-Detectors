import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_CSV = Path(r"D:\HaitongNi\MEngProject\repos\ReStraV\results\genvidbench_common1000\restrav_genvidbench_common1000_predictions.csv")
OUT_CSV = Path(r"D:\HaitongNi\MEngProject\repos\ReStraV\results\genvidbench_common1000\restrav_genvidbench_common1000_generator_breakdown.csv")

df = pd.read_csv(PRED_CSV)

print("Columns:", df.columns.tolist())

if "y_true" in df.columns:
    y_col = "y_true"
elif "label" in df.columns:
    y_col = "label"
else:
    raise RuntimeError("Cannot find label column.")

if "prob_fake" in df.columns:
    score_col = "prob_fake"
elif "score_fake" in df.columns:
    score_col = "score_fake"
else:
    raise RuntimeError("Cannot find fake score column.")

if "pred_label" not in df.columns:
    raise RuntimeError("Cannot find pred_label column.")

df["y_true"] = df[y_col].astype(int)
df["score_fake"] = df[score_col].astype(float)
df["pred_label"] = df["pred_label"].astype(int)

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
        "fake_recall_at_train_threshold": (g["pred_label"] == 1).mean(),
        "mean_score_fake": g["score_fake"].mean(),
        "auc_vs_all_real": auc,
        "ap_vs_all_real": ap,
    })

out = pd.DataFrame(rows).sort_values("fake_recall_at_train_threshold")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("Saved:", OUT_CSV)
print()
print(out.to_string(index=False))

print("\nCheck:")
print("rows:", len(df))
print("real:", len(real))
print("fake:", len(fake))
print("missing generator_id:", df["generator_id"].isna().sum())


Per-generator breakdown:
Note: ReStraV was evaluated as a trained-on-subset baseline. The generator breakdown is calculated only on the held-out 500-sample test split, which contains 250 real and 250 fake samples. The threshold was selected on the training split: best_tau = 0.6450.

Metric definitions:
fake_recall_at_train_threshold = proportion of fake videos from this generator classified as fake using the threshold selected on the training set
mean_score_fake = average predicted fake probability for this generator
auc_vs_all_real = AUC using this generator's fake test samples vs all 250 real test samples
ap_vs_all_real = AP using this generator's fake test samples vs all 250 real test samples

Harder generators for ReStraV:
pika      n=29  recall=0.6552  mean_score=0.7195  AUC_vs_real=0.9527  AP_vs_real=0.7607
mora      n=28  recall=0.7500  mean_score=0.7661  AUC_vs_real=0.9444  AP_vs_real=0.8221

Generators with very high fake recall:
ms        n=27  recall=1.0000  mean_score=0.9967  AUC_vs_real=1.0000  AP_vs_real=1.0000
cogvideo  n=31  recall=1.0000  mean_score=0.9992  AUC_vs_real=1.0000  AP_vs_real=1.0000
musev     n=36  recall=1.0000  mean_score=0.9358  AUC_vs_real=0.9952  AP_vs_real=0.9673
svd       n=37  recall=1.0000  mean_score=0.9893  AUC_vs_real=1.0000  AP_vs_real=1.0000
t2vz      n=29  recall=1.0000  mean_score=0.9959  AUC_vs_real=1.0000  AP_vs_real=1.0000
vc2       n=33  recall=1.0000  mean_score=0.9916  AUC_vs_real=1.0000  AP_vs_real=1.0000

Observation:
ReStraV performs very strongly across most GenVidBench fake generators after training on the target subset. Six out of eight generators achieve 100% fake recall on the held-out test split. The relatively harder generators are pika and mora, where recall drops to 0.6552 and 0.7500, respectively. However, their AUC values are still high, above 0.94, suggesting that ReStraV still ranks these fake samples well but the train-selected threshold is less optimal for them.

Because this is a trained-on-subset result, the generator-wise performance should be interpreted as within-benchmark adaptation rather than pure zero-shot generalization.