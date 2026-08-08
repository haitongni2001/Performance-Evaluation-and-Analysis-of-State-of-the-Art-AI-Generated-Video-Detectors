import os
import re
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def extract_answer(text: str) -> str:
    """
    Extract answer from <answer>...</answer>.
    If no tag exists, return the raw text.
    """
    if text is None:
        return ""

    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text.strip()


def normalize_choice(ans: str):
    """
    Convert model output to A/B.
    A = AI-generated / fake
    B = real
    """
    if ans is None:
        return None

    ans = ans.strip().upper()

    # Direct answers
    if ans == "A":
        return "A"
    if ans == "B":
        return "B"

    # Common verbose forms
    if ans.startswith("A"):
        return "A"
    if ans.startswith("B"):
        return "B"

    if "AI-GENERATED" in ans or "AI GENERATED" in ans or "FAKE" in ans or "SYNTHETIC" in ans:
        return "A"

    if "REAL" in ans or "AUTHENTIC" in ans:
        return "B"

    return None


def build_question(sample):
    """
    Build the same style of prompt as VidGuard-R1 eval_bench.py.
    """
    if sample["problem_type"] == "multiple choice":
        question = sample["problem"] + "\nOptions:\n"
        for op in sample["options"]:
            question += op + "\n"
    else:
        question = sample["problem"]

    question_template = (
        "{Question}\n"
        "Please think about this question carefully. "
        "Provide your final answer between the <answer> and </answer> tags. "
        "For this multiple-choice question, provide only the single option letter "
        "(A or B) within the <answer> </answer> tags."
    )

    return question_template.format(Question=question)


def make_message(sample, fps=1.0, max_pixels=224 * 224):
    """
    Qwen2.5-VL message format.
    Lower fps and max_pixels reduce memory usage.
    """
    video_path = sample["path"]

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {
                    "type": "text",
                    "text": build_question(sample),
                },
            ],
        }
    ]

    return message


def load_existing_output(output_path):
    if not output_path.exists():
        return [], 0

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        results = existing.get("results", [])
        return results, len(results)
    except Exception as e:
        print(f"Could not read existing output file: {e}")
        return [], 0


def save_output(output_path, results, final_acc=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obj = {"results": results}
    if final_acc is not None:
        obj["final_acc"] = [final_acc]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="VidGuard-R1 evaluation using transformers instead of vLLM.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_pixels", type=int, default=336 * 336)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples is not None:
        data = data[: args.max_samples]

    print(f"Loaded samples: {len(data)}")
    print(f"Model: {args.model_path}")
    print(f"Output: {output_path}")

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    results, start_idx = load_existing_output(output_path)
    if start_idx > 0:
        print(f"Resuming from sample index {start_idx}")

    for idx in tqdm(range(start_idx, len(data)), desc="Evaluating"):
        sample = data[idx]

        try:
            message = make_message(
                sample,
                fps=args.fps,
                max_pixels=args.max_pixels,
            )

            text = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                message,
                return_video_kwargs=True,
            )

            # qwen-vl-utils may return values like {"fps": [0.97]}.
            # For single-sample inference, transformers processor expects scalar fps, not a list.
            clean_video_kwargs = {}
            for k, v in video_kwargs.items():
                if isinstance(v, list) and len(v) == 1:
                    clean_video_kwargs[k] = v[0]
                else:
                    clean_video_kwargs[k] = v

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **clean_video_kwargs,
            )

            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            # Remove prompt tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        except Exception as e:
            print(f"\nError at index {idx}, path={sample.get('path')}")
            print(e)
            output_text = "<answer>error</answer>"

        pred_text = extract_answer(output_text)
        pred_choice = normalize_choice(pred_text)

        gt_text = extract_answer(sample.get("solution", ""))
        gt_choice = normalize_choice(gt_text)

        # A = fake = 1, B = real = 0
        if pred_choice == "A":
            pred_label = 1
        elif pred_choice == "B":
            pred_label = 0
        else:
            pred_label = None

        if gt_choice == "A":
            gt_label = 1
        elif gt_choice == "B":
            gt_label = 0
        else:
            gt_label = None

        correct = (pred_label == gt_label) if pred_label is not None and gt_label is not None else False

        out_sample = dict(sample)
        out_sample["output"] = output_text
        out_sample["prediction_text"] = pred_text
        out_sample["prediction_choice"] = pred_choice
        out_sample["prediction_label"] = pred_label
        out_sample["gt_choice"] = gt_choice
        out_sample["gt_label"] = gt_label
        out_sample["correct"] = correct

        results.append(out_sample)

        if (idx + 1) % args.save_every == 0:
            save_output(output_path, results)

    # Compute metrics only for valid A/B predictions
    y_true = []
    y_pred = []

    invalid = 0
    for r in results:
        if r.get("gt_label") is None or r.get("prediction_label") is None:
            invalid += 1
            continue
        y_true.append(int(r["gt_label"]))
        y_pred.append(int(r["prediction_label"]))

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()
    else:
        acc = precision = recall = f1 = mcc = 0.0
        cm = [[0, 0], [0, 0]]

    final_acc = {
        "valid_samples": len(y_true),
        "invalid_predictions": invalid,
        "accuracy": acc,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "mcc": mcc,
        "confusion_matrix": cm,
        "note": "A/fake is treated as positive class 1; B/real is treated as class 0.",
    }

    save_output(output_path, results, final_acc=final_acc)

    print("\n" + "=" * 60)
    print("Final metrics")
    print("=" * 60)
    print(f"Valid samples: {len(y_true)}")
    print(f"Invalid predictions: {invalid}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision_fake: {precision:.4f}")
    print(f"Recall_fake: {recall:.4f}")
    print(f"F1_fake: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]: {cm}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()