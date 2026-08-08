import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.model_selection import train_test_split
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

import dinov2_features as d2


# =========================
# Paths
# =========================

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000_all_opensource.csv")

RESULT_DIR = Path(r"D:\HaitongNi\MEngProject\results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CACHE = RESULT_DIR / "restrav_aigvd_all_opensource_1000_features.npz"
META_CACHE = RESULT_DIR / "restrav_aigvd_all_opensource_1000_metadata.csv"

PRED_CSV = RESULT_DIR / "restrav_aigvd_all_opensource_1000_predictions.csv"
RESULT_TXT = RESULT_DIR / "restrav_aigvd_all_opensource_1000_results.txt"

MODEL_OUT = RESULT_DIR / "restrav_aigvd_all_opensource_1000_model.pt"
MEAN_OUT = RESULT_DIR / "restrav_aigvd_all_opensource_1000_mean.npy"
STD_OUT = RESULT_DIR / "restrav_aigvd_all_opensource_1000_std.npy"
TAU_OUT = RESULT_DIR / "restrav_aigvd_all_opensource_1000_best_tau.npy"


# =========================
# Settings
# =========================

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ReStraV default idea: short clip, 24 frames, around 2 seconds
NUM_FRAMES = 24
WINDOW_SEC = 2.0

TEST_SIZE = 0.5
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3


# =========================
# Model
# =========================

class MLP(nn.Module):
    def __init__(self, in_dim=21, h1=64, h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_or_load_features():
    """
    Extract ReStraV 21-D features for all videos in INPUT_CSV.
    If cached features exist, load them directly.
    """
    if FEATURE_CACHE.exists() and META_CACHE.exists():
        print("Loading cached features:")
        print(FEATURE_CACHE)
        data = np.load(FEATURE_CACHE)
        X = data["features"].astype(np.float32)
        y = data["labels"].astype(np.int64)
        meta = pd.read_csv(META_CACHE)
        return X, y, meta

    df = pd.read_csv(INPUT_CSV)

    features = []
    labels = []
    meta_rows = []
    failed_rows = []

    print("Extracting ReStraV features...")
    print("Input samples:", len(df))
    print("Device:", DEVICE)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        video_path = row["path"]

        try:
            with torch.no_grad():
                Z = d2.extract_dinov2_embeddings(
                    [video_path],
                    device=DEVICE,
                    T=NUM_FRAMES,
                    window_sec=WINDOW_SEC,
                )
                feat = d2.features_from_Z(Z)

            feat_np = feat.detach().cpu().numpy().astype(np.float32)[0]

            # Our AIGVDBench label convention:
            # real = 0, fake = 1
            label = int(row["label"])

            features.append(feat_np)
            labels.append(label)

            meta_rows.append({
                "orig_index": idx,
                "video_id": row.get("video_id", ""),
                "path": video_path,
                "label": label,
                "label_name": row.get("label_name", ""),
                "source_type": row.get("source_type", ""),
                "task_type": row.get("task_type", ""),
                "generator_id": row.get("generator_id", ""),
                "split": row.get("split", ""),
                "error": "",
            })

        except Exception as e:
            err = str(e)
            print("\nFeature extraction failed:")
            print(video_path)
            print(err)

            failed_rows.append({
                "orig_index": idx,
                "video_id": row.get("video_id", ""),
                "path": video_path,
                "label": int(row["label"]),
                "label_name": row.get("label_name", ""),
                "source_type": row.get("source_type", ""),
                "task_type": row.get("task_type", ""),
                "generator_id": row.get("generator_id", ""),
                "split": row.get("split", ""),
                "error": err,
            })

        # Incremental cache every 25 successful samples
        if len(features) > 0 and len(features) % 25 == 0:
            X_tmp = np.stack(features, axis=0).astype(np.float32)
            y_tmp = np.array(labels, dtype=np.int64)
            meta_tmp = pd.DataFrame(meta_rows)

            np.savez_compressed(FEATURE_CACHE, features=X_tmp, labels=y_tmp)
            meta_tmp.to_csv(META_CACHE, index=False, encoding="utf-8-sig")

    if len(features) == 0:
        raise RuntimeError("No features were extracted successfully.")

    X = np.stack(features, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    np.savez_compressed(FEATURE_CACHE, features=X, labels=y)
    meta.to_csv(META_CACHE, index=False, encoding="utf-8-sig")

    if failed_rows:
        failed_csv = RESULT_DIR / "restrav_aigvd_all_opensource_1000_failed.csv"
        pd.DataFrame(failed_rows).to_csv(failed_csv, index=False, encoding="utf-8-sig")
        print("Failed samples saved to:", failed_csv)

    print("Saved feature cache:", FEATURE_CACHE)
    print("Saved metadata cache:", META_CACHE)
    print("Valid features:", len(X))
    print("Failed:", len(failed_rows))

    return X, y, meta


def make_loader(X, y, batch_size=32, shuffle=False):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def train_model(X_train, y_train):
    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)

    model = MLP(in_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)
            yb = yb.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} loss: {avg_loss:.4f}")

    return model


def predict_probs(model, X):
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64), BATCH_SIZE, shuffle=False)

    model.eval()
    probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            probs.append(out)

    return np.concatenate(probs)


def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 181)

    best_tau = 0.5
    best_f1 = -1.0

    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    return best_tau, best_f1


def main():
    set_seed(SEED)

    X, y, meta = extract_or_load_features()

    print("\nFeature matrix:", X.shape)
    print("Label distribution:", pd.Series(y).value_counts().to_dict())
    print("Positive class: fake = 1, real = 0")

    # Stratified split.
    # Note: This makes ReStraV a trained-on-subset baseline.
    idx = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEED,
    )

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    # Normalize using train statistics only.
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    np.save(MEAN_OUT, mean)
    np.save(STD_OUT, std)

    print("\nTrain samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("Train label distribution:", pd.Series(y_train).value_counts().to_dict())
    print("Test label distribution:", pd.Series(y_test).value_counts().to_dict())

    model = train_model(X_train, y_train)

    train_probs = predict_probs(model, X_train)
    best_tau, train_best_f1 = find_best_threshold(y_train, train_probs)

    print(f"\nBest threshold on train: {best_tau:.4f}")
    print(f"Best train F1: {train_best_f1:.4f}")

    test_probs = predict_probs(model, X_test)
    test_preds = (test_probs >= best_tau).astype(int)

    acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    mcc = matthews_corrcoef(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    ap = average_precision_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)

    # Save model and tau
    torch.save(model.state_dict(), MODEL_OUT)
    np.save(TAU_OUT, best_tau)

    pred_df = meta_test.copy()
    pred_df["y_true"] = y_test
    pred_df["prob_fake"] = test_probs
    pred_df["pred_label"] = test_preds
    pred_df["correct"] = pred_df["y_true"] == pred_df["pred_label"]
    pred_df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    result = f"""ReStraV Evaluation Results
Implementation: DINOv2 temporal geometry feature extraction + MLP training
Dataset: AIGVDBench all-open-source common-1000
Input CSV: {INPUT_CSV}
Feature cache: {FEATURE_CACHE}

Valid samples: {len(X)}
Train samples: {len(X_train)}
Test samples: {len(X_test)}
Positive class: fake = 1, real = 0

Feature:
21-D temporal geometry vector
NUM_FRAMES = {NUM_FRAMES}
WINDOW_SEC = {WINDOW_SEC}

Training:
MLP hidden sizes: 64, 32
Epochs: {EPOCHS}
Learning rate: {LR}
Train/test split: stratified 50/50
Normalization: train-set mean/std only
Best threshold selected on train set: {best_tau:.4f}

Threshold-free metrics on test set:
ROC_AUC_fake: {auc:.4f}
AP_fake: {ap:.4f}

Threshold-based metrics on test set:
Accuracy: {acc:.4f}
Precision_fake: {precision:.4f}
Recall_fake: {recall:.4f}
F1_fake: {f1:.4f}
MCC: {mcc:.4f}

Confusion Matrix [[TN, FP], [FN, TP]]:
{cm}

Prediction CSV: {PRED_CSV}
Model: {MODEL_OUT}
Mean: {MEAN_OUT}
Std: {STD_OUT}
Best threshold: {TAU_OUT}

Important note:
ReStraV requires training a lightweight classifier on extracted temporal geometry features.
Therefore, this result is a trained-on-subset evaluation rather than pure pretrained inference.
"""

    RESULT_TXT.write_text(result, encoding="utf-8")

    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()