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

wave_path = r"D:\HaitongNi\MEngProject\repos\WaveRep-SyntheticVideoDetection\results\aigvd_common1000\waverep_aigvd_all_opensource_1000_predictions.csv"

aigvdet_path = r"D:\HaitongNi\MEngProject\repos\AIGVDet\results\aigvd_common1000\aigvdet_aigvd_common1000_predictions.csv"

out_csv = r"D:\HaitongNi\MEngProject\ensemble\alpha_sweep_results.csv"

wave = pd.read_csv(wave_path)
aigv = pd.read_csv(aigvdet_path)

merged = wave.merge(
    aigv[
        [
            "video_id",
            "generator_id",
            "final_score"
        ]
    ],
    on=[
        "video_id",
        "generator_id"
    ],
    how="inner"
)

y_true = merged["label"].astype(int)

rows = []

for alpha in [x / 100 for x in range(0, 101, 5)]:

    score = (
        alpha * merged["prob_fake"]
        + (1 - alpha) * merged["final_score"]
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