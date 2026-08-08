import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ALPHA = 0.70

wave_path = r"D:\HaitongNi\MEngProject_SecondBench\01_WaveRep\results\waverep_genvidbench_common1000_predictions.csv"

aigvdet_path = r"D:\HaitongNi\MEngProject\repos\AIGVDet\results\genvidbench_common1000\aigvdet_genvidbench_common1000_predictions.csv"

out_csv = r"D:\HaitongNi\MEngProject\ensemble\ensemble_genvidbench_generator_breakdown_alpha070.csv"

# --------------------------------------------------
# Load
# --------------------------------------------------

wave = pd.read_csv(wave_path)
aigv = pd.read_csv(aigvdet_path)

print("WaveRep rows:", len(wave))
print("AIGVDet rows:", len(aigv))

# --------------------------------------------------
# Build merge key
# --------------------------------------------------

wave["merge_key"] = (
    wave["video_id"]
    .astype(str)
    .str.replace("/", "\\", regex=False)
)

aigv["merge_key"] = (
    aigv["path"]
    .astype(str)
    .str.extract(r"(Pair[12].*)", expand=False)
    .str.replace("/", "\\", regex=False)
)

# --------------------------------------------------
# Merge
# --------------------------------------------------

df = wave.merge(
    aigv[
        [
            "merge_key",
            "final_score"
        ]
    ],
    on="merge_key",
    how="inner"
)

print("Merged rows:", len(df))

# --------------------------------------------------
# Ensemble score
# --------------------------------------------------

df["ensemble_score"] = (
    ALPHA * df["prob_fake"]
    + (1.0 - ALPHA) * df["final_score"]
)

# --------------------------------------------------
# Split real/fake
# --------------------------------------------------

real_df = df[df["label"] == 0].copy()
fake_df = df[df["label"] == 1].copy()

print("Real:", len(real_df))
print("Fake:", len(fake_df))

# --------------------------------------------------
# Generator breakdown
# --------------------------------------------------

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
print("Summary:")
print("Worst generator:")
print(result.iloc[0])

print()
print("Best generator:")
print(result.iloc[-1])