import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from natsort import natsorted

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

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

sys.path.append("core")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils1.utils import get_network, str2bool


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000_all_opensource.csv",
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default=r"D:\HaitongNi\MEngProject\aigvdet_cache\aigvd_common1000",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"D:\HaitongNi\MEngProject\results\aigvdet_aigvd_common1000_predictions.csv",
    )

    parser.add_argument(
        "--result_txt",
        type=str,
        default=r"D:\HaitongNi\MEngProject\results\aigvdet_aigvd_common1000_results.txt",
    )

    parser.add_argument(
        "--generator_csv",
        type=str,
        default=r"D:\HaitongNi\MEngProject\results\aigvdet_aigvd_common1000_generator_breakdown.csv",
    )

    parser.add_argument("--raft_model", type=str, default=r"raft_model\raft-things.pth")
    parser.add_argument("--model_optical", type=str, default=r"checkpoints\optical_aug.pth")
    parser.add_argument("--model_original", type=str, default=r"checkpoints\original_aug.pth")

    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=-1)

    # Important: use a fixed number of frames for feasible common-1000 evaluation.
    # Set -1 to use all frames, but that can be very slow.
    parser.add_argument("--num_frames", type=int, default=16)

    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--alternate_corr", action="store_true")

    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--aug_norm", type=str2bool, default=True)

    return parser.parse_args()


def safe_name(label_name, idx):
    if label_name == "real":
        return f"real_{idx:05d}"
    return f"fake_{idx:05d}"


def load_image_for_raft(imfile, device):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def extract_rgb_frames(video_path, output_folder, num_frames=16):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    existing = sorted(list(output_folder.glob("*.png")))
    if num_frames > 0 and len(existing) >= num_frames:
        return natsorted([str(p) for p in existing[:num_frames]])
    if num_frames < 0 and len(existing) > 0:
        return natsorted([str(p) for p in existing])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    if num_frames > 0:
        frame_indices = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        frame_indices = np.arange(total)

    saved = []
    last_frame = None

    for out_idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            if last_frame is None:
                continue
            frame = last_frame
        else:
            last_frame = frame

        out_path = output_folder / f"frame_{out_idx:05d}.png"
        cv2.imwrite(str(out_path), frame)
        saved.append(str(out_path))

    cap.release()

    if not saved:
        raise RuntimeError(f"Could not decode any frame: {video_path}")

    return natsorted(saved)


def build_raft(args, device):
    raft = torch.nn.DataParallel(RAFT(args))
    state = torch.load(args.raft_model, map_location=device)
    raft.load_state_dict(state)

    raft = raft.module
    raft.to(device)
    raft.eval()
    return raft


def generate_optical_flow_images(raft, rgb_frames, flow_folder, device):
    flow_folder = Path(flow_folder)
    flow_folder.mkdir(parents=True, exist_ok=True)

    expected = max(0, len(rgb_frames) - 1)
    existing = sorted(list(flow_folder.glob("*.png")))
    if len(existing) >= expected and expected > 0:
        return natsorted([str(p) for p in existing[:expected]])

    flow_paths = []

    with torch.no_grad():
        for i, (im1, im2) in enumerate(zip(rgb_frames[:-1], rgb_frames[1:])):
            out_path = flow_folder / f"flow_{i:05d}.png"

            if out_path.exists():
                flow_paths.append(str(out_path))
                continue

            image1 = load_image_for_raft(im1, device)
            image2 = load_image_for_raft(im2, device)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = raft(image1, image2, iters=20, test_mode=True)

            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flo_img = flow_viz.flow_to_image(flo)

            cv2.imwrite(str(out_path), flo_img)
            flow_paths.append(str(out_path))

    return natsorted(flow_paths)


def load_detector(checkpoint_path, arch, device):
    model = get_network(arch)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def score_images(model, image_paths, device, aug_norm=True):
    if len(image_paths) == 0:
        return np.nan

    trans = transforms.Compose(
        (
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
        )
    )

    probs = []

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img = trans(img)

        if aug_norm:
            img = TF.normalize(
                img,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        x = img.unsqueeze(0).to(device)

        with torch.no_grad():
            prob = model(x).sigmoid().item()

        probs.append(prob)

    return float(np.mean(probs))


def compute_metrics(df, threshold=0.5):
    valid = df.dropna(subset=["final_score"]).copy()

    y_true = valid["label"].astype(int).values
    y_score = valid["final_score"].astype(float).values
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
    valid["correct"] = valid["pred_label"] == valid["label"].astype(int)

    return metrics, valid


def generator_breakdown(pred_df):
    rows = []

    real = pred_df[pred_df["label"] == 0].copy()
    fake = pred_df[pred_df["label"] == 1].copy()

    for gen, g in fake.groupby("generator_id"):
        pair = pd.concat([real, g], ignore_index=True)

        auc = np.nan
        ap = np.nan

        if pair["label"].nunique() == 2:
            auc = roc_auc_score(pair["label"], pair["final_score"])
            ap = average_precision_score(pair["label"], pair["final_score"])

        rows.append(
            {
                "generator_id": gen,
                "n_fake": len(g),
                "fake_recall_at_0.5": (g["final_score"] >= 0.5).mean(),
                "mean_final_score_fake": g["final_score"].mean(),
                "mean_original_score_fake": g["original_score"].mean(),
                "mean_optical_score_fake": g["optical_score"].mean(),
                "auc_vs_all_real": auc,
                "ap_vs_all_real": ap,
            }
        )

    return pd.DataFrame(rows).sort_values("fake_recall_at_0.5")


def main():
    args = parse_args()

    device = torch.device("cpu" if args.use_cpu else DEVICE)
    print("Device:", device)

    for required in [args.raft_model, args.model_optical, args.model_original]:
        if not Path(required).exists():
            raise FileNotFoundError(required)

    df = pd.read_csv(args.input_csv)

    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    work_dir = Path(args.work_dir)
    rgb_root = work_dir / "rgb"
    flow_root = work_dir / "flow"

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.generator_csv).parent.mkdir(parents=True, exist_ok=True)

    print("Loading RAFT...")
    raft = build_raft(args, device)

    print("Loading AIGVDet detectors...")
    model_original = load_detector(args.model_original, args.arch, device)
    model_optical = load_detector(args.model_optical, args.arch, device)

    results = []

    real_count = 0
    fake_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="AIGVDet inference"):
        label = int(row["label"])
        label_name = str(row["label_name"])

        if label == 0:
            real_count += 1
            sample_id = safe_name("real", real_count)
            class_dir = "0_real"
        else:
            fake_count += 1
            sample_id = safe_name("fake", fake_count)
            class_dir = "1_fake"

        rgb_dir = rgb_root / class_dir / sample_id
        flow_dir = flow_root / class_dir / sample_id

        try:
            rgb_frames = extract_rgb_frames(
                row["path"],
                rgb_dir,
                num_frames=args.num_frames,
            )

            flow_frames = generate_optical_flow_images(
                raft,
                rgb_frames,
                flow_dir,
                device,
            )

            original_score = score_images(
                model_original,
                rgb_frames,
                device,
                aug_norm=args.aug_norm,
            )

            optical_score = score_images(
                model_optical,
                flow_frames,
                device,
                aug_norm=args.aug_norm,
            )

            final_score = 0.5 * original_score + 0.5 * optical_score

            error = ""

        except Exception as e:
            original_score = np.nan
            optical_score = np.nan
            final_score = np.nan
            error = str(e)
            print("Failed:", row["path"], error)

        results.append(
            {
                "sample_id": sample_id,
                "video_id": row.get("video_id", ""),
                "path": row["path"],
                "label": label,
                "label_name": label_name,
                "source_type": row.get("source_type", ""),
                "task_type": row.get("task_type", ""),
                "generator_id": row.get("generator_id", ""),
                "original_score": original_score,
                "optical_score": optical_score,
                "final_score": final_score,
                "error": error,
            }
        )

        # incremental save
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    pred_raw = pd.DataFrame(results)
    pred_raw.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    metrics, pred = compute_metrics(pred_raw, threshold=args.threshold)
    pred.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    gen_df = generator_breakdown(pred)
    gen_df.to_csv(args.generator_csv, index=False, encoding="utf-8-sig")

    result = f"""AIGVDet Evaluation Results
Implementation: pretrained inference with RGB branch + optical-flow branch
Dataset: AIGVDBench all-open-source common subset
Input CSV: {args.input_csv}
Samples requested: {len(df)}
Valid samples: {metrics["valid_samples"]}

Weights:
Original branch: {args.model_original}
Optical branch: {args.model_optical}
RAFT: {args.raft_model}

Frame setting:
num_frames per video: {args.num_frames}
final_score = 0.5 * original_score + 0.5 * optical_score

Threshold-free metrics:
ROC_AUC_fake: {metrics["roc_auc_fake"]:.6f}
AP_fake: {metrics["ap_fake"]:.6f}

Threshold-based metrics @ final_score >= {args.threshold}:
Accuracy: {metrics["accuracy"]:.6f}
Precision_fake: {metrics["precision_fake"]:.6f}
Recall_fake: {metrics["recall_fake"]:.6f}
F1_fake: {metrics["f1_fake"]:.6f}
MCC: {metrics["mcc"]:.6f}

Confusion Matrix [[TN, FP], [FN, TP]]:
{metrics["confusion_matrix"]}

Prediction CSV:
{args.output_csv}

Generator breakdown CSV:
{args.generator_csv}
"""

    Path(args.result_txt).write_text(result, encoding="utf-8")

    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()