import pandas as pd

p = r"D:\HaitongNi\MEngProject\repos\AIGVDet\results\genvidbench_common1000\aigvdet_genvidbench_common1000_predictions.csv"

df = pd.read_csv(p)

print(df.columns.tolist())

print()
print(df.head(10).to_string())

print()
print(df[["sample_id"]].head(20).to_string())

print()
print("rows:", len(df))