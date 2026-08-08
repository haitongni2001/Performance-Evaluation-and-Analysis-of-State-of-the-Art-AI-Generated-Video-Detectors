import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\01_WaveRep\results\waverep_genvidbench_common1000_predictions.csv")
META_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\metadata\genvidbench_common1000.csv")

OUT_CSV = Path(r"D:\HaitongNi\MEngProject_SecondBench\01_WaveRep\results\waverep_genvidbench_common1000_generator_breakdown.csv")

pred = pd.read_csv(PRED_CSV)
meta = pd.read_csv(META_CSV)

print("Prediction columns:", pred.columns.tolist())
print("Metadata columns:", meta.columns.tolist())

# Normalize path keys
def norm_path(x):
    return str(x).replace("\\", "/").lower()

# Try common columns from our WaveRep prediction script
if "path" in pred.columns:
    pred["join_key"] = pred["path"].apply(norm_path)
else:
    raise RuntimeError("Prediction CSV does not contain path column.")

meta["join_key"] = meta["path"].apply(norm_path)

df = pred.merge(meta, on="join_key", how="left", suffixes=("", "_meta"))

# Label
if "label" in df.columns:
    y_col = "label"
elif "actual_label" in df.columns:
    y_col = "actual_label"
else:
    raise RuntimeError("Could not find label column.")

# Fake score
if "prob_fake" in df.columns:
    score_col = "prob_fake"
elif "score_fake" in df.columns:
    score_col = "score_fake"
elif "pred_prob_fake" in df.columns:
    score_col = "pred_prob_fake"
else:
    raise RuntimeError("Could not find fake probability column.")

df["y_true"] = df[y_col].astype(int)
df["score_fake"] = df[score_col].astype(float)
df["pred_label"] = (df["score_fake"] >= 0.5).astype(int)

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

print("\nMerge check:")
print("pred rows:", len(pred))
print("metadata rows:", len(meta))
print("merged rows:", len(df))
print("missing generator_id:", df["generator_id"].isna().sum())

Per-generator breakdown:
Note: WaveRep was evaluated on GenVidBench common-1000 using pretrained inference. Each fake generator has 62 or 63 fake samples in the subset.

Metric definitions:
fake_recall_at_0.5 = proportion of fake videos from this generator classified as fake at threshold 0.5
mean_score_fake = average predicted fake probability for this generator
auc_vs_all_real = AUC using this generator's fake samples vs all 500 real samples
ap_vs_all_real = AP using this generator's fake samples vs all 500 real samples

Harder generator for WaveRep:
ms        n=63  recall=0.6984  mean_score=0.6980  AUC_vs_real=0.9695  AP_vs_real=0.8336

Generators with high fake recall:
svd       n=62  recall=0.9355  mean_score=0.9451  AUC_vs_real=0.9958  AP_vs_real=0.9768
t2vz      n=62  recall=0.9839  mean_score=0.9771  AUC_vs_real=0.9984  AP_vs_real=0.9895
mora      n=63  recall=0.9841  mean_score=0.9881  AUC_vs_real=0.9993  AP_vs_real=0.9955
musev     n=63  recall=1.0000  mean_score=0.9983  AUC_vs_real=0.9999  AP_vs_real=0.9993
cogvideo  n=63  recall=1.0000  mean_score=0.9983  AUC_vs_real=0.9999  AP_vs_real=0.9993
pika      n=62  recall=1.0000  mean_score=0.9973  AUC_vs_real=0.9998  AP_vs_real=0.9985
vc2       n=62  recall=1.0000  mean_score=0.9941  AUC_vs_real=0.9996  AP_vs_real=0.9973

Observation:
WaveRep performs very strongly across almost all GenVidBench fake generator categories. Most generators achieve near-perfect or perfect fake recall at the default threshold 0.5, with AUC values close to 1.0 against all real samples. The only relatively harder generator is ms, where fake recall drops to 0.6984. However, even for ms, the AUC remains high at 0.9695, suggesting that WaveRep still ranks fake videos above real videos well, but the default threshold is less well calibrated for this generator.

Compared with D3, WaveRep is much more stable across generators. D3 almost fails on t2vz, while WaveRep achieves 0.9839 recall and 0.9984 AUC on t2vz. This supports the observation that WaveRep provides stronger and more generator-robust pretrained inference on GenVidBench.
