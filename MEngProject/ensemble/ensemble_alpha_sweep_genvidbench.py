import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

wave_path = r"D:\HaitongNi\MEngProject_SecondBench\01_WaveRep\results\waverep_genvidbench_common1000_predictions.csv"

aigvdet_path = r"D:\HaitongNi\MEngProject\repos\AIGVDet\results\genvidbench_common1000\aigvdet_genvidbench_common1000_predictions.csv"

out_csv = r"D:\HaitongNi\MEngProject\ensemble\alpha_sweep_results_genvidbench.csv"

# --------------------------------------------------
# Load
# --------------------------------------------------

wave = pd.read_csv(wave_path)
aigv = pd.read_csv(aigvdet_path)

print("WaveRep rows:", len(wave))
print("AIGVDet rows:", len(aigv))

# --------------------------------------------------
# Build merge key
# --------------------------------------------------

wave["merge_key"] = (
    wave["video_id"]
    .astype(str)
    .str.replace("/", "\\", regex=False)
)

aigv["merge_key"] = (
    aigv["path"]
    .astype(str)
    .str.extract(r"(Pair[12].*)", expand=False)
    .str.replace("/", "\\", regex=False)
)

# --------------------------------------------------
# Merge
# --------------------------------------------------

merged = wave.merge(
    aigv[
        [
            "merge_key",
            "final_score"
        ]
    ],
    on="merge_key",
    how="inner"
)

print()
print("Merged rows:", len(merged))
print()

print(
    merged[
        [
            "merge_key",
            "label",
            "prob_fake",
            "final_score"
        ]
    ].head()
)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

y_true = merged["label"].astype(int)

rows = []

for alpha in [x / 100 for x in range(0, 101, 5)]:

    score = (
        alpha * merged["prob_fake"]
        + (1.0 - alpha) * merged["final_score"]
    )

    pred = (score >= 0.5).astype(int)

    rows.append(
        {
            "alpha": alpha,
            "auc": roc_auc_score(y_true, score),
            "ap": average_precision_score(y_true, score),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "mcc": matthews_corrcoef(y_true, pred),
        }
    )

result = pd.DataFrame(rows)

result.to_csv(out_csv, index=False)

print()
print(result.round(6).to_string(index=False))

print()
print("Saved:")
print(out_csv)

best_row = result.loc[result["mcc"].idxmax()]

print()
print("Best MCC:")
print(best_row)