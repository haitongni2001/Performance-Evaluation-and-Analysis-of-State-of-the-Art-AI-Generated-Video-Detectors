import os
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
import pandas as pd
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

from options.test_options import TestOption
from utils.trainer import Trainer
from utils.utils import get_logger, set_random_seed, get_customized_test_dataset_configs
from dataset import get_test_dataloader
from builder import get_model


AIGVD_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000_all_opensource.csv")

OUT_PRED_CSV = Path(r"D:\HaitongNi\MEngProject\results\mmdet_aigvd_common1000_predictions.csv")
OUT_RESULT_TXT = Path(r"D:\HaitongNi\MEngProject\results\mmdet_aigvd_common1000_results.txt")
OUT_GENERATOR_CSV = Path(r"D:\HaitongNi\MEngProject\results\mmdet_aigvd_common1000_generator_breakdown.csv")


def to_scalar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.numpy().tolist()

    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x.tolist()

    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return to_scalar(x[0])
        return [to_scalar(v) for v in x]

    return x


def normalize_sample_id(x):
    """
    Expected MM-Det video ids look like:
      real_00371
      fake_00023
    Sometimes keys may include path fragments or frame suffixes.
    """
    x = str(x).replace("\\", "/")
    x = os.path.basename(x)

    if "__" in x:
        x = x.split("__")[0]

    if x.lower().endswith(".jpg"):
        x = x[:-4]
    if x.lower().endswith(".mp4"):
        x = x[:-4]

    return x


def build_metadata_mapping():
    """
    Reconstruct mapping from MM-Det copied filenames back to AIGVDBench metadata.
    This follows prepare_aigvd_1000_for_mmdet.py:
      real_00001.mp4 ... real_00500.mp4
      fake_00001.mp4 ... fake_00500.mp4
    """
    df = pd.read_csv(AIGVD_CSV)

    rows = []

    real = df[df["label_name"] == "real"].reset_index(drop=True)
    fake = df[df["label_name"] == "fake"].reset_index(drop=True)

    for i, (_, row) in enumerate(real.iterrows(), start=1):
        r = row.to_dict()
        r["mmdet_id"] = f"real_{i:05d}"
        rows.append(r)

    for i, (_, row) in enumerate(fake.iterrows(), start=1):
        r = row.to_dict()
        r["mmdet_id"] = f"fake_{i:05d}"
        rows.append(r)

    meta = pd.DataFrame(rows)
    return meta


def collect_results_from_validation(trainer, dataloader, stop_count=-1):
    """
    Similar to trainer.validation_video(), but saves sample-level outputs.
    """
    all_gt = {}
    all_pred = {}
    all_proba = {}

    for step, batch in enumerate(tqdm(dataloader, desc="Validation savepred")):
        if stop_count is not None and stop_count > 0 and step >= stop_count:
            break

        results = trainer.validation_step(batch)

        for k, v in results.get("gt", {}).items():
            all_gt[normalize_sample_id(k)] = to_scalar(v)

        for k, v in results.get("pred", {}).items():
            all_pred[normalize_sample_id(k)] = to_scalar(v)

        for k, v in results.get("proba", {}).items():
            all_proba[normalize_sample_id(k)] = to_scalar(v)

    keys = sorted(set(all_gt.keys()) | set(all_pred.keys()) | set(all_proba.keys()))

    rows = []
    for k in keys:
        rows.append({
            "mmdet_id": k,
            "y_true_raw": all_gt.get(k, None),
            "pred_raw": all_pred.get(k, None),
            "score_raw": all_proba.get(k, None),
        })

    return pd.DataFrame(rows)


def coerce_label(x):
    """
    Try to convert MM-Det gt/pred into 0/1.
    Expected: 0_real -> 0, 1_fake -> 1.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    if isinstance(x, str):
        xs = x.lower()
        if "fake" in xs or xs == "1" or xs == "1_fake":
            return 1
        if "real" in xs or xs == "0" or xs == "0_real":
            return 0

    try:
        val = float(x)
        return int(round(val))
    except Exception:
        return np.nan


def coerce_score(x):
    """
    Convert score/proba to float.
    If MM-Det returns a list like [p_real, p_fake], use p_fake.
    """
    if x is None:
        return np.nan

    if isinstance(x, str):
        try:
            # handle strings like "[0.2, 0.8]"
            import ast
            obj = ast.literal_eval(x)
            return coerce_score(obj)
        except Exception:
            try:
                return float(x)
            except Exception:
                return np.nan

    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return np.nan
        if len(x) >= 2:
            return float(x[1])
        return float(x[0])

    try:
        return float(x)
    except Exception:
        return np.nan


def compute_metrics(df, score_col="score_fake", threshold=0.5):
    valid = df.dropna(subset=["y_true", score_col]).copy()

    y_true = valid["y_true"].astype(int).values
    y_score = valid[score_col].astype(float).values
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "valid_samples": len(valid),
        "threshold": threshold,
        "roc_auc_fake": roc_auc_score(y_true, y_score),
        "ap_fake": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    valid["pred_label"] = y_pred
    valid["correct"] = valid["pred_label"] == valid["y_true"]

    return metrics, valid


def generator_breakdown(pred_df):
    """
    For fake-only generator groups:
      - recall
      - mean_score

    Also compute generator-vs-all-real AUC:
      each generator's fake samples vs all real samples.
    """
    rows = []
    real = pred_df[pred_df["y_true"] == 0].copy()

    fake = pred_df[pred_df["y_true"] == 1].copy()

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
            "fake_recall_at_0.5": ((g["score_fake"] >= 0.5).astype(int) == 1).mean(),
            "mean_score_fake": g["score_fake"].mean(),
            "auc_vs_all_real": auc,
            "ap_vs_all_real": ap,
        })

    out = pd.DataFrame(rows).sort_values("fake_recall_at_0.5")
    return out


def main():
    args = TestOption().parse()
    config = args.__dict__

    logger = get_logger(__name__, config)
    logger.info(config)

    set_random_seed(config["seed"])

    dataset_classes = config["classes"]
    logger.info(f"Validation on {dataset_classes}.")

    test_dataset_configs = get_customized_test_dataset_configs(config)

    # Same as test_customized.py
    config["st_pretrained"] = False
    config["st_ckpt"] = None

    model = get_model(config)
    model.eval()

    path = None
    if os.path.exists(config["ckpt"]):
        logger.info(f'Load checkpoint from {config["ckpt"]}')
        path = config["ckpt"]
    elif os.path.exists(os.path.join("expts", config["expt"], "checkpoints")):
        if os.path.exists(os.path.join("expts", config["expt"], "checkpoints", "current_model_best.pth")):
            path = os.path.join("expts", config["expt"], "checkpoints", "current_model_best.pth")
        elif os.path.exists(os.path.join("expts", config["expt"], "checkpoints", "current_model_latest.pth")):
            path = os.path.join("expts", config["expt"], "checkpoints", "current_model_latest.pth")

    if path is None:
        raise ValueError(f'Checkpoint not found: {config["ckpt"]}')

    state_dict = torch.load(path)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=config["cache_mm"])

    # Important for our custom save-prediction loop:
    # ensure model weights and input tensors are on the same device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_pred_dfs = []

    for dataset_class, test_dataset_config in zip(dataset_classes, test_dataset_configs):
        test_config = deepcopy(config)
        test_config["datasets"] = test_dataset_config

        trainer = Trainer(
            config=test_config,
            model=model,
            logger=logger,
        )

        trainer.val_dataloader = get_test_dataloader(test_dataset_config)

        stop_count = config.get("sample_size", -1)

        pred_df = collect_results_from_validation(
            trainer,
            trainer.val_dataloader,
            stop_count=stop_count,
        )

        pred_df["dataset_class"] = dataset_class
        all_pred_dfs.append(pred_df)

    pred_raw = pd.concat(all_pred_dfs, ignore_index=True)

    pred_raw["y_true"] = pred_raw["y_true_raw"].apply(coerce_label)
    pred_raw["pred_from_repo"] = pred_raw["pred_raw"].apply(coerce_label)
    pred_raw["score_fake"] = pred_raw["score_raw"].apply(coerce_score)

    meta = build_metadata_mapping()
    pred = pred_raw.merge(meta, on="mmdet_id", how="left", suffixes=("", "_meta"))

    # If score direction is opposite, report both.
    metrics, pred_valid = compute_metrics(pred, score_col="score_fake", threshold=0.5)

    pred["score_fake_reversed"] = 1.0 - pred["score_fake"]
    metrics_rev, _ = compute_metrics(pred, score_col="score_fake_reversed", threshold=0.5)

    # Choose the direction with higher AUC for final reporting.
    # This avoids ambiguity if MM-Det proba is p(real) instead of p(fake).
    if metrics_rev["roc_auc_fake"] > metrics["roc_auc_fake"]:
        pred["score_fake"] = pred["score_fake_reversed"]
        metrics, pred_valid = compute_metrics(pred, score_col="score_fake", threshold=0.5)
        score_note = "Used reversed score because 1-score gave higher ROC-AUC."
    else:
        score_note = "Used original score as fake score."

    OUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)

    pred_valid.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")

    gen_df = generator_breakdown(pred_valid)
    gen_df.to_csv(OUT_GENERATOR_CSV, index=False, encoding="utf-8-sig")

    result = f"""MM-Det Evaluation Results
Implementation: official customized-dataset pipeline with saved sample-level predictions
Dataset: AIGVDBench all-open-source common-1000
Samples: 500 real + 500 fake
Checkpoint: {config["ckpt"]}
Cache MM: {config["cache_mm"]}

Score note:
{score_note}

Valid samples: {metrics["valid_samples"]}
Threshold: {metrics["threshold"]}

Threshold-free metrics:
ROC_AUC_fake: {metrics["roc_auc_fake"]:.6f}
AP_fake: {metrics["ap_fake"]:.6f}

Threshold-based metrics @ score_fake >= 0.5:
Accuracy: {metrics["accuracy"]:.6f}
Precision_fake: {metrics["precision_fake"]:.6f}
Recall_fake: {metrics["recall_fake"]:.6f}
F1_fake: {metrics["f1_fake"]:.6f}
MCC: {metrics["mcc"]:.6f}

Confusion Matrix [[TN, FP], [FN, TP]]:
{metrics["confusion_matrix"]}

Prediction CSV:
{OUT_PRED_CSV}

Generator breakdown CSV:
{OUT_GENERATOR_CSV}
"""

    OUT_RESULT_TXT.write_text(result, encoding="utf-8")

    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()