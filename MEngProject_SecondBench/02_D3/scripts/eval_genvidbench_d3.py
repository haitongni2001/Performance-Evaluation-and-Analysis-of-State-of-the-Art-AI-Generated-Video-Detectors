import os
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import datetime
import pandas as pd
from pathlib import Path

D3_REPO_DIR = Path(r"D:\HaitongNi\MEngProject\repos\D3")
sys.path.insert(0, str(D3_REPO_DIR))

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from data import D3_dataset_AP
from models import D3_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_best_threshold(y_true, fake_score):
    """
    Find the threshold that gives the best F1 score on the current set.
    This is useful for diagnostic comparison, but should not be presented
    as a strictly fair test metric unless threshold is chosen on validation set.
    """
    thresholds = np.unique(fake_score)
    best = {
        "threshold": None,
        "accuracy": -1,
        "precision": -1,
        "recall": -1,
        "f1": -1,
        "confusion_matrix": None,
    }

    for th in thresholds:
        y_pred = (fake_score >= th).astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        if f1 > best["f1"]:
            best["threshold"] = th
            best["accuracy"] = acc
            best["precision"] = precision
            best["recall"] = recall
            best["f1"] = f1
            best["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return best


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="D3 evaluation with extended metrics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--loss", type=str, default="l2", choices=["l2", "cos"])
    parser.add_argument(
        "--encoder",
        type=str,
        default="XCLIP-16",
        choices=[
            "CLIP-16",
            "CLIP-32",
            "XCLIP-16",
            "XCLIP-32",
            "DINO-base",
            "DINO-large",
            "ResNet-18",
            "VGG-16",
            "EfficientNet-b4",
            "MobileNet-v3",
        ],
    )
    parser.add_argument("--real-csv", type=str, default=None)
    parser.add_argument("--fake-csv", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=999999)
    args = parser.parse_args()

    seed_everything(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(f"Starting evaluation for {args.encoder} with {args.loss} loss")
    print(f"Real CSV: {args.real_csv}")
    print(f"Fake CSV: {args.fake_csv}")
    print(f"Max len per class: {args.max_len}")

    # Load model
    model = D3_model(encoder_type=args.encoder, loss_type=args.loss).cuda()
    model.eval()

    # Load dataset
    eval_dataset = D3_dataset_AP(
        real_csv=args.real_csv,
        fake_csv=args.fake_csv,
        max_len=args.max_len,
    )

    print(f"Total samples: {len(eval_dataset)}")

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    # Evaluation
    y_true = []
    raw_score = []

    with torch.no_grad():
        for batch_frames, batch_label in tqdm(eval_loader, desc="Evaluating"):
            batch_inputs = batch_frames.cuda()
            _, _, batch_dis_std = model(batch_inputs)

            raw_score.extend(batch_dis_std.cpu().flatten().numpy())
            y_true.extend(batch_label.cpu().flatten().numpy())

    y_true = np.array(y_true).astype(int)
    raw_score = np.array(raw_score)

    # In original D3 code, AP was computed as average_precision_score(1-y_true, raw_score).
    # So raw_score is treated as a real-positive score.
    real_score = raw_score

    # For fake detection, we reverse the score direction.
    fake_score = -raw_score

    # Threshold-free metrics
    ap_real = average_precision_score(1 - y_true, real_score)
    ap_fake = average_precision_score(y_true, fake_score)

    auc_real = roc_auc_score(1 - y_true, real_score)
    auc_fake = roc_auc_score(y_true, fake_score)

    # Best threshold diagnostics on current set
    best = find_best_threshold(y_true, fake_score)

    # Save per-sample predictions
    real_df = pd.read_csv(args.real_csv).head(args.max_len)
    fake_df = pd.read_csv(args.fake_csv).head(args.max_len)
    meta_df = pd.concat([real_df, fake_df], axis=0, ignore_index=True)

    meta_df["y_true"] = y_true
    meta_df["raw_score_real_direction"] = raw_score
    meta_df["fake_score"] = fake_score
    meta_df["pred_best_threshold"] = (fake_score >= best["threshold"]).astype(int)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    pred_csv = f"results/predictions_{timestamp}.csv"
    meta_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    result_str = (
        f"D3 Evaluation Results\n"
        f"Encoder: {args.encoder}\n"
        f"Loss Type: {args.loss}\n"
        f"Real CSV: {args.real_csv}\n"
        f"Fake CSV: {args.fake_csv}\n"
        f"Total Samples: {len(y_true)}\n\n"
        f"Threshold-free metrics:\n"
        f"AP_fake: {ap_fake:.4f}\n"
        f"AP_real: {ap_real:.4f}\n"
        f"ROC_AUC_fake: {auc_fake:.4f}\n"
        f"ROC_AUC_real: {auc_real:.4f}\n\n"
        f"Best-threshold diagnostic metrics, selected on this same test subset:\n"
        f"Best Threshold on fake_score: {best['threshold']:.6f}\n"
        f"Best Accuracy: {best['accuracy']:.4f}\n"
        f"Best Precision_fake: {best['precision']:.4f}\n"
        f"Best Recall_fake: {best['recall']:.4f}\n"
        f"Best F1_fake: {best['f1']:.4f}\n"
        f"Confusion Matrix [[TN, FP], [FN, TP]]:\n"
        f"{best['confusion_matrix']}\n\n"
        f"Prediction CSV: {pred_csv}\n"
    )

    print("\n" + "=" * 60)
    print(result_str.strip())
    print("=" * 60)

    output_file = f"results/result_{timestamp}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result_str)

    print(f"\nResults saved to {output_file}")
    print(f"Predictions saved to {pred_csv}")