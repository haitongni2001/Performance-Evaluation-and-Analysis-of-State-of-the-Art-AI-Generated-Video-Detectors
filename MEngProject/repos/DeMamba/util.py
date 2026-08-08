import models
import time
import torch
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, average_precision_score, roc_auc_score

def build_model(model_name):
    if model_name == 'F3Net':
        model = models.Det_F3_Net()
    if model_name == 'NPR':
        model = models.resnet50_npr()
    if model_name == 'STIL':
        model = models.Det_STIL()
    if model_name == 'XCLIP_DeMamba':
        model = models.XCLIP_DeMamba()
    if model_name == 'CLIP_DeMamba':
        model = models.CLIP_DeMamba()
    if model_name == 'XCLIP':
        model = models.XCLIP()
    if model_name == 'CLIP':
        model = models.CLIP_Base()
    if model_name == 'ViT_B_MINTIME':
        model = models.ViT_B_MINTIME()
    return model

def eval_model(cfg, model, val_loader, loss_ce, val_batch_size):
    model.eval()
    outpred_list = []
    gt_label_list = []
    video_list = []
    valLoss = 0
    lossTrainNorm = 0
    print("******** Start Testing. ********")

    with torch.no_grad():  # No need to track gradients during validation
        for i, (_, input, target, binary_label, video_id) in enumerate(tqdm(val_loader, desc="Validation", total=len(val_loader))):
            if i == 0:
                ss_time = time.time()
            
            input = input[:,0]
            varInput = torch.autograd.Variable(input.float().cuda())
            varTarget = torch.autograd.Variable(target.contiguous().cuda())
            var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())

            logit = model(varInput)
            lossvalue = loss_ce(logit, var_Binary_Target)

            valLoss += lossvalue.item()
            lossTrainNorm += 1
            outpred_list.append(logit[:,0].sigmoid().cpu().detach().numpy())
            gt_label_list.append(varTarget.cpu().detach().numpy())
            video_list.append(video_id)
    
    valLoss = valLoss / lossTrainNorm

    outpred = np.concatenate(outpred_list, 0)
    gt_label = np.concatenate(gt_label_list, 0)
    video_list = np.concatenate(video_list, 0)
    pred_labels = [1 if item > 0.5 else 0 for item in outpred]
    true_labels = np.argmax(gt_label, axis=1)

    pred_accuracy = accuracy_score(true_labels, pred_labels)

    return pred_accuracy, video_list, pred_labels, true_labels, outpred

def train_one_epoch(cfg, model, loss_ce, scheduler, optimizer, epochID, max_epoch, max_acc, train_loader, val_loader, snapshot_path):
    model.train()
    trainLoss = 0
    lossTrainNorm = 0

    scheduler.step()
    pbar = tqdm(total=cfg['bath_per_epoch'])
    for batchID, (index, input, target, binary_label) in enumerate(train_loader):
        if batchID > cfg['bath_per_epoch']:
            break
        if batchID == 0:
            ss_time = time.time()
        input = input[:,0].float()
        varInput = torch.autograd.Variable(input).cuda()
        varTarget = torch.autograd.Variable(target.contiguous().cuda())
        var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())
        optimizer.zero_grad()

        logit = model(varInput)
        lossvalue = loss_ce(logit, var_Binary_Target)
        
        lossvalue.backward()
        optimizer.step()

        trainLoss += lossvalue.item()
        lossTrainNorm += 1
        pbar.set_postfix(loss=trainLoss / lossTrainNorm)
        pbar.update(1)
        del lossvalue

    trainLoss = trainLoss / lossTrainNorm
    
    if (epochID+1) % 1 == 0:
        pred_accuracy, video_id, pred_labels, true_labels, outpred = eval_model(cfg, model, val_loader, loss_ce, cfg['val_batch_size'])    

        torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
            snapshot_path + "/last"+ ".pth",
            )

        if pred_accuracy > max_acc:
            max_epoch, max_acc = epochID, pred_accuracy
            torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
            snapshot_path + "/best_acc"+ ".pth",
            )

        df_result = pd.DataFrame({
            'data_path': video_id,
            'predicted_label': pred_labels,
            'actual_label': true_labels,
            'predicted_prob':outpred
        })

        temp_result_txt = snapshot_path+'/Epoch_'+str(epochID)+'_accuracy.txt'
        with open(temp_result_txt, 'w') as file:
            true_labels = df_result['actual_label']
            pred_probs = df_result['predicted_prob'] 
            auc = roc_auc_score(true_labels, pred_probs)
            ap = average_precision_score(true_labels, pred_probs)
            file.write(f"总正确率: {pred_accuracy:.2%}\n")
            file.write(f"AUC是: {auc:.2%}\n")
            file.write(f"AP是: {ap:.2%}\n")

        prefixes = ["fake/ModelScope", "fake/Morph", "fake/MoonValley", 
                    "fake/HotShot", "fake/EvalCrafter_T2V_Dataset/show_1", 
                    "fake/Sora", "fake/Wild", "fake/VideoCrafter",
                   "fake/Lavies", "fake/Gen2"]

        video_nums = [700, 700, 626, 700, 700, 56, 926, 1400, 1400, 1380]

        # real 
        condition = df_result['data_path'].apply(lambda x: x.startswith("real"))
        temp_df_val = df_result[condition]
        temp_df_val['correct'] = temp_df_val['predicted_label'] == temp_df_val['actual_label']
        accuracy = temp_df_val['correct'].mean()

        FP = int((1-accuracy) * 10000) if not np.isnan(accuracy) else 0 if not np.isnan(accuracy) else 0 if not np.isnan(accuracy) else 0 if not np.isnan(accuracy) else 0

        for index, temp_prefixes in enumerate(prefixes):
            condition = df_result['data_path'].apply(lambda x: x.startswith(temp_prefixes))
            temp_df_val = df_result[condition]
            temp_df_val['correct'] = temp_df_val['predicted_label'] == temp_df_val['actual_label']
            accuracy = temp_df_val['correct'].mean()

            TP = int(accuracy * video_nums[index]) if not np.isnan(accuracy) else 0 if not np.isnan(accuracy) else 0 if not np.isnan(accuracy) else 0
            FN = int((1-accuracy) * video_nums[index]) if not np.isnan(accuracy) else 0 if not np.isnan(accuracy) else 0
            P = TP / (TP + FP) if (TP + FP) > 0 else 0
            R = TP / (TP + FN) if (TP + FN) > 0 else 0
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

            condition |= df_result['data_path'].str.startswith('real')
            temp_df_val = df_result[condition]
            true_labels = temp_df_val['actual_label']
            pred_probs = temp_df_val['predicted_prob']  # 假设这是模型预测的概率
            ap = average_precision_score(true_labels, pred_probs)
            with open(temp_result_txt, 'a') as file:
                name = temp_prefixes.split('/')[-1]
                file.write(f"文件名: {name}, Recall是: {accuracy}\n")
                file.write(f"文件名: {name}, F1是: {F1}\n")
                file.write(f"文件名: {name}, AP是: {ap}\n")

        print("*****Average Training loss",str(trainLoss),"*****\n")
        print("*****Epoch", str(epochID), "*****Acc ", str(pred_accuracy), '*****',
            '\n', "*****Max acc epoch", str(max_epoch), "*****Acc ", str(max_acc), '*****\n')
    end_time = time.time()

    return max_epoch, max_acc, end_time - ss_time


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

        # DeMamba dataloader returns [B, 1, T, C, H, W].
        # Model expects [B, T, C, H, W].
        if input.dim() == 6 and input.shape[1] == 1:
            input = input.squeeze(1)

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
