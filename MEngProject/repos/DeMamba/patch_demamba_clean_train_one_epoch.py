from pathlib import Path

p = Path("util.py")
s = p.read_text(encoding="utf-8")

append_code = r'''

# ============================================================
# Clean AIGVDBench-compatible train_one_epoch override
# This overrides the original GenVideo-specific train_one_epoch.
# It keeps training + overall validation metrics, but removes
# hard-coded GenVideo subgroup statistics that cause NaN / empty-set crashes.
# ============================================================

def train_one_epoch(cfg, model, loss_ce, scheduler, optimizer, epochID, max_epoch, max_acc, train_loader, val_loader, snapshot_path):
    import os
    import time
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from sklearn.metrics import (
        accuracy_score,
        recall_score,
        precision_score,
        f1_score,
        average_precision_score,
        roc_auc_score,
        matthews_corrcoef,
        confusion_matrix,
    )

    start_time = time.time()

    model.train()
    trainLoss = 0.0
    lossTrainNorm = 0

    pbar = tqdm(train_loader, total=cfg.get("bath_per_epoch", len(train_loader)))

    for batchID, (index, input, target, binary_label) in enumerate(pbar):
        if batchID > cfg.get("bath_per_epoch", len(train_loader)):
            break

        optimizer.zero_grad()

        varInput = torch.autograd.Variable(input.contiguous().cuda())
        var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())

        logit = model(varInput)
        lossvalue = loss_ce(logit, var_Binary_Target)

        lossvalue.backward()
        optimizer.step()

        trainLoss += lossvalue.item()
        lossTrainNorm += 1

        pbar.set_postfix(loss=trainLoss / max(lossTrainNorm, 1))

        del lossvalue

    scheduler.step()

    trainLoss = trainLoss / max(lossTrainNorm, 1)

    # Overall validation only. No GenVideo-specific subgroup statistics.
    pred_accuracy, video_id, pred_labels, true_labels, outpred = eval_model(
        cfg, model, val_loader, loss_ce, cfg["val_batch_size"]
    )

    # Flatten video_id list if it is a list of batches/tuples.
    flat_video_ids = []
    for item in video_id:
        if isinstance(item, (list, tuple)):
            flat_video_ids.extend(list(item))
        else:
            flat_video_ids.append(item)

    true_labels = np.asarray(true_labels).astype(int)
    pred_labels = np.asarray(pred_labels).astype(int)
    pred_probs = np.asarray(outpred).astype(float)

    # Safety checks.
    n = min(len(flat_video_ids), len(true_labels), len(pred_labels), len(pred_probs))
    flat_video_ids = flat_video_ids[:n]
    true_labels = true_labels[:n]
    pred_labels = pred_labels[:n]
    pred_probs = pred_probs[:n]

    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels).tolist()

    if len(np.unique(true_labels)) == 2:
        auc = roc_auc_score(true_labels, pred_probs)
        ap = average_precision_score(true_labels, pred_probs)
    else:
        auc = float("nan")
        ap = float("nan")

    os.makedirs(snapshot_path, exist_ok=True)

    # Save checkpoints.
    torch.save(
        {
            "epoch": epochID,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": trainLoss,
            "val_accuracy": acc,
            "val_auc": auc,
            "val_ap": ap,
        },
        os.path.join(snapshot_path, "last.pth"),
    )

    if acc > max_acc:
        max_epoch, max_acc = epochID, acc
        torch.save(
            {
                "epoch": epochID,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": trainLoss,
                "val_accuracy": acc,
                "val_auc": auc,
                "val_ap": ap,
            },
            os.path.join(snapshot_path, "best_acc.pth"),
        )

    # Save per-sample predictions.
    df_result = pd.DataFrame(
        {
            "data_path": flat_video_ids,
            "actual_label": true_labels,
            "predicted_label": pred_labels,
            "pred_prob_fake": pred_probs,
        }
    )

    pred_csv = os.path.join(snapshot_path, f"Epoch_{epochID}_predictions.csv")
    df_result.to_csv(pred_csv, index=False)

    # Save readable metrics.
    result_txt = os.path.join(snapshot_path, f"Epoch_{epochID}_accuracy.txt")
    with open(result_txt, "w", encoding="utf-8") as file:
        file.write(f"Epoch: {epochID}\n")
        file.write(f"Train Loss: {trainLoss:.6f}\n")
        file.write(f"Accuracy: {acc:.6f}\n")
        file.write(f"Precision_fake: {precision:.6f}\n")
        file.write(f"Recall_fake: {recall:.6f}\n")
        file.write(f"F1_fake: {f1:.6f}\n")
        file.write(f"MCC: {mcc:.6f}\n")
        file.write(f"ROC_AUC_fake: {auc:.6f}\n")
        file.write(f"AP_fake: {ap:.6f}\n")
        file.write(f"Confusion Matrix [[TN, FP], [FN, TP]]:\n{cm}\n")
        file.write(f"Prediction CSV: {pred_csv}\n")

    print("******* AIGVDBench validation results *******")
    print(f"Epoch: {epochID}")
    print(f"Train Loss: {trainLoss:.6f}")
    print(f"Accuracy: {acc:.6f}")
    print(f"Precision_fake: {precision:.6f}")
    print(f"Recall_fake: {recall:.6f}")
    print(f"F1_fake: {f1:.6f}")
    print(f"MCC: {mcc:.6f}")
    print(f"ROC_AUC_fake: {auc:.6f}")
    print(f"AP_fake: {ap:.6f}")
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]: {cm}")

    epoch_time = time.time() - start_time
    return max_epoch, max_acc, epoch_time
'''

# Append override only once.
marker = "# Clean AIGVDBench-compatible train_one_epoch override"
if marker not in s:
    s = s + append_code
else:
    # If already appended, replace from marker onward.
    idx = s.index("# ============================================================\n# Clean AIGVDBench-compatible train_one_epoch override")
    s = s[:idx] + append_code

p.write_text(s, encoding="utf-8")
print("appended clean train_one_epoch override to util.py")