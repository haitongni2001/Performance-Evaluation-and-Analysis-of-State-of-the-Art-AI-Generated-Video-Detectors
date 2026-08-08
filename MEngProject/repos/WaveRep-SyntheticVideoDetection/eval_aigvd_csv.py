import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demo"
sys.path.insert(0, str(DEMO_DIR))

import torch
import pandas as pd
import numpy as np
from scipy.special import expit
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

from utils import create_transform, create_model, ReadVideoIteratorCV, evaluate

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000_all_opensource.csv")
OUTPUT_CSV = Path(r"D:\HaitongNi\MEngProject\results\waverep_aigvd_all_opensource_1000_predictions.csv")
OUTPUT_TXT = Path(r"D:\HaitongNi\MEngProject\results\waverep_aigvd_all_opensource_1000_results.txt")

WEIGHTS = Path(r"D:\HaitongNi\MEngProject\repos\WaveRep-SyntheticVideoDetection\demo\weights\weights_dinov2_G4.ckpt")

DEVICE = "cuda:0"
LIMIT = None
CROPPING = 504
ARC = "vit_base_patch14_reg4_dinov2.lvd142m"


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = create_transform(CROPPING)
    model = create_model(str(WEIGHTS), ARC, CROPPING, device)

    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="WaveRep AIGVD"):
        video_path = row["path"]

        try:
            data_loader = ReadVideoIteratorCV(video_path, transform=transform, limit=LIMIT)
            tab = evaluate(model, data_loader, device)

            if LIMIT:
                score_logit = float(tab["logit"].iloc[:LIMIT].mean())
            else:
                score_logit = float(tab["logit"].mean())

            score_prob = float(expit(score_logit))
            error = ""

        except Exception as e:
            score_logit = np.nan
            score_prob = np.nan
            error = str(e)
            print("\nError:", video_path)
            print(error)

        rows.append({
            "video_id": row["video_id"],
            "path": video_path,
            "label": int(row["label"]),
            "label_name": row["label_name"],
            "source_type": row["source_type"],
            "task_type": row["task_type"],
            "generator_id": row["generator_id"],
            "logit": score_logit,
            "prob_fake": score_prob,
            "error": error,
        })

        if (idx + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    valid = out.dropna(subset=["prob_fake"]).copy()
    y_true = valid["label"].astype(int).values
    y_score = valid["prob_fake"].astype(float).values
    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)

    result = f"""WaveRep Evaluation Results
Dataset: AIGVDBench all-open-source common-1000
Weights: {WEIGHTS}
Valid samples: {len(valid)}
Invalid samples: {len(out) - len(valid)}

Threshold-free metrics:
ROC_AUC_fake: {auc:.4f}
AP_fake: {ap:.4f}

Threshold @ prob_fake >= 0.5:
Accuracy: {acc:.4f}
Precision_fake: {precision:.4f}
Recall_fake: {recall:.4f}
F1_fake: {f1:.4f}
MCC: {mcc:.4f}
Confusion Matrix [[TN, FP], [FN, TP]]:
{cm}
Prediction CSV: {OUTPUT_CSV}
"""

    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)

    OUTPUT_TXT.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()