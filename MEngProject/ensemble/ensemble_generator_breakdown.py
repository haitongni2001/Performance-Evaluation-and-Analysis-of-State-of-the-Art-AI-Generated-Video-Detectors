import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ALPHA = 0.90

wave_path = r"D:\HaitongNi\MEngProject\repos\WaveRep-SyntheticVideoDetection\results\aigvd_common1000\waverep_aigvd_all_opensource_1000_predictions.csv"

aigvdet_path = r"D:\HaitongNi\MEngProject\repos\AIGVDet\results\aigvd_common1000\aigvdet_aigvd_common1000_predictions.csv"

out_csv = r"D:\HaitongNi\MEngProject\ensemble\ensemble_aigvd_generator_breakdown_alpha090.csv"

wave = pd.read_csv(wave_path)
aigv = pd.read_csv(aigvdet_path)

df = wave.merge(
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

df["ensemble_score"] = (
    ALPHA * df["prob_fake"]
    + (1.0 - ALPHA) * df["final_score"]
)

real_df = df[df["label"] == 0].copy()
fake_df = df[df["label"] == 1].copy()

rows = []

for gid, group in fake_df.groupby("generator_id"):

    scores_fake = group["ensemble_score"]

    recall = (
        (scores_fake >= 0.5)
        .astype(int)
        .mean()
    )

    binary_df = pd.concat(
        [
            real_df,
            group
        ],
        ignore_index=True
    )

    auc = roc_auc_score(
        binary_df["label"],
        binary_df["ensemble_score"]
    )

    ap = average_precision_score(
        binary_df["label"],
        binary_df["ensemble_score"]
    )

    rows.append(
        {
            "generator_id": gid,
            "n_fake": len(group),
            "fake_recall_at_0.5": recall,
            "mean_ensemble_score": scores_fake.mean(),
            "auc_vs_all_real": auc,
            "ap_vs_all_real": ap,
        }
    )

result = pd.DataFrame(rows)

result = result.sort_values(
    "fake_recall_at_0.5"
).reset_index(drop=True)

result.to_csv(out_csv, index=False)

print()
print("Generator breakdown:")
print(result.round(6).to_string(index=False))

print()
print("Saved:")
print(out_csv)

print()
print("Check:")
print("rows:", len(df))
print("real:", (df['label'] == 0).sum())
print("fake:", (df['label'] == 1).sum())
print("missing generator_id:", df['generator_id'].isna().sum())