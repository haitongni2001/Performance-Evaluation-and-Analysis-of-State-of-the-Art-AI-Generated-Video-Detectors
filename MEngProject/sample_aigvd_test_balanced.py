import pandas as pd
from pathlib import Path

INPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_index.csv")
OUTPUT_CSV = Path(r"F:\MEngDatasets\AIGVDBench\metadata\aigvd_test_balanced_6000.csv")

df = pd.read_csv(INPUT_CSV)

real_df = df[df["label_name"] == "real"].copy()
fake_df = df[df["label_name"] == "fake"].copy()

N_REAL = 3000
N_FAKE_TOTAL = 3000

# 1. real 抽 3000 个，刚好全用
real_sample = real_df.sample(
    n=min(N_REAL, len(real_df)),
    random_state=42
)

# 2. fake 按 generator_id 平均抽样
generator_ids = sorted(fake_df["generator_id"].dropna().unique())
n_generators = len(generator_ids)

per_generator = N_FAKE_TOTAL // n_generators
remainder = N_FAKE_TOTAL % n_generators

fake_samples = []

for i, gen_id in enumerate(generator_ids):
    gen_df = fake_df[fake_df["generator_id"] == gen_id]

    n = per_generator + (1 if i < remainder else 0)
    n = min(n, len(gen_df))

    sample = gen_df.sample(n=n, random_state=42)
    fake_samples.append(sample)

fake_sample = pd.concat(fake_samples, ignore_index=True)

# 3. 合并并打乱
subset = pd.concat([real_sample, fake_sample], ignore_index=True)
subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. 保存
subset.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved subset to: {OUTPUT_CSV}")
print(f"Total size: {len(subset)}")

print("\nLabel counts:")
print(subset["label_name"].value_counts())

print("\nSource type counts:")
print(subset["source_type"].value_counts())

print("\nTask type counts:")
print(subset["task_type"].value_counts())

print("\nGenerator ID counts:")
print(subset["generator_id"].value_counts())