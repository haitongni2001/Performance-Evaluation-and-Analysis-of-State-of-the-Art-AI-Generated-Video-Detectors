import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

wave_path = r"D:\HaitongNi\MEngProject\repos\WaveRep-SyntheticVideoDetection\results\aigvd_common1000\waverep_aigvd_all_opensource_1000_predictions.csv"

aigvdet_path = r"D:\HaitongNi\MEngProject\repos\AIGVDet\results\aigvd_common1000\aigvdet_aigvd_common1000_predictions.csv"

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

print("Merged rows:", len(merged))

alpha = 0.90

merged["ensemble_score"] = (
    alpha * merged["prob_fake"]
    + (1.0 - alpha) * merged["final_score"]
)

y_true = merged["label"].astype(int)
y_score = merged["ensemble_score"]

y_pred = (y_score >= 0.5).astype(int)

auc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)

print()
print("Alpha =", alpha)
print()

print(f"ROC_AUC_fake = {auc:.6f}")
print(f"AP_fake = {ap:.6f}")

print(f"Accuracy = {acc:.6f}")
print(f"Precision_fake = {prec:.6f}")
print(f"Recall_fake = {rec:.6f}")
print(f"F1_fake = {f1:.6f}")
print(f"MCC = {mcc:.6f}")

print()
print("Confusion Matrix:")
print(cm)