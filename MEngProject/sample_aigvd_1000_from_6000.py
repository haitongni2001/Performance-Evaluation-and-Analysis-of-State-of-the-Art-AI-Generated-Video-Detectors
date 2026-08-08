import pandas as pd
from pathlib import Path

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_6000.csv")
OUTPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_1000.csv")

df = pd.read_csv(INPUT_CSV)

real_df = df[df["label_name"] == "real"].copy()
fake_df = df[df["label_name"] == "fake"].copy()

# 500 real
real_sample = real_df.sample(n=500, random_state=42)

# 500 fake, evenly across generator_id
fake_samples = []
generator_ids = sorted(fake_df["generator_id"].unique())

per_gen = 500 // len(generator_ids)
remainder = 500 % len(generator_ids)

for i, gen_id in enumerate(generator_ids):
    gen_df = fake_df[fake_df["generator_id"] == gen_id]
    n = per_gen + (1 if i < remainder else 0)
    fake_samples.append(gen_df.sample(n=n, random_state=42))

fake_sample = pd.concat(fake_samples, ignore_index=True)

subset = pd.concat([real_sample, fake_sample], ignore_index=True)
subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)

subset.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_CSV)
print("Total:", len(subset))

print("\nLabel counts:")
print(subset["label_name"].value_counts())

print("\nGenerator counts:")
print(subset["generator_id"].value_counts())