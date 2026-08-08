from pathlib import Path

p = Path("test_dMMD.py")
s = p.read_text(encoding="utf-8")

# 1) Add per-sample CSV saving after each dataset evaluation.
old_main_block = '''            logger.info(
                f"Dataset: {name} | Recall: {test_results['recall']:.4f} | F1: {test_results['f1']:.4f} | "
                f"Accuracy: {test_results['accuracy']:.4f} | Precision: {test_results['precision']:.4f} | "
                f"AUROC: {test_results['auroc']:.4f}"
            )
'''

new_main_block = '''            logger.info(
                f"Dataset: {name} | Recall: {test_results['recall']:.4f} | F1: {test_results['f1']:.4f} | "
                f"Accuracy: {test_results['accuracy']:.4f} | Precision: {test_results['precision']:.4f} | "
                f"AUROC: {test_results['auroc']:.4f}"
            )

            # Save per-sample scores for generator-wise analysis.
            real_ids_path = os.path.join(cfg.data.data_path, "split", "real", cfg.data.test_real_model, "test_ids.txt")
            fake_ids_path = os.path.join(cfg.data.data_path, "split", "fake", name, "test_ids.txt")

            with open(real_ids_path, "r", encoding="utf-8") as f:
                real_ids = [line.strip() for line in f if line.strip()]
            with open(fake_ids_path, "r", encoding="utf-8") as f:
                fake_ids = [line.strip() for line in f if line.strip()]

            n_real = len(test_results["dt_clean"])
            n_fake = len(test_results["dt_adv"])

            pred_rows = []
            for i in range(n_real):
                pred_rows.append({
                    "sample_id": real_ids[i] if i < len(real_ids) else f"real_{i+1:05d}.mp4",
                    "label": 0,
                    "label_name": "real",
                    "raw_score": float(test_results["dt_clean"][i]),
                    "pred_label_at_1": int(float(test_results["dt_clean"][i]) > 1),
                    "generator_id": cfg.data.test_real_model,
                })

            for i in range(n_fake):
                sample_id = fake_ids[i] if i < len(fake_ids) else f"fake_{i+1:05d}.mp4"
                # Our GenVidBench copied filenames are like fake_00001_generator.mp4
                stem = os.path.splitext(sample_id)[0]
                parts = stem.split("_")
                gen = "_".join(parts[2:]) if len(parts) >= 3 else name

                pred_rows.append({
                    "sample_id": sample_id,
                    "label": 1,
                    "label_name": "fake",
                    "raw_score": float(test_results["dt_adv"][i]),
                    "pred_label_at_1": int(float(test_results["dt_adv"][i]) > 1),
                    "generator_id": gen,
                })

            pred_csv_path = os.path.join(cfg.log_path, "nsgvd_genvidbench_common1000_predictions.csv")
            os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)
            pd.DataFrame(pred_rows).to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
            logger.success(f"Per-sample predictions saved to {pred_csv_path}")
'''

if old_main_block not in s:
    raise RuntimeError("Could not find main logger block to patch.")

s = s.replace(old_main_block, new_main_block)

# 2) Add raw scores to return dict.
old_return = '''    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
    }
'''

new_return = '''    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
        "dt_clean": dt_clean.cpu().numpy(),
        "dt_adv": dt_adv.cpu().numpy(),
        "raw_predict": raw_predict.cpu().numpy(),
        "predict": predict.cpu().numpy() if hasattr(predict, "cpu") else predict,
        "labels": labels.cpu().numpy(),
    }
'''

if old_return not in s:
    raise RuntimeError("Could not find return block to patch.")

s = s.replace(old_return, new_return)

p.write_text(s, encoding="utf-8")
print("patched test_dMMD.py to save per-sample predictions")