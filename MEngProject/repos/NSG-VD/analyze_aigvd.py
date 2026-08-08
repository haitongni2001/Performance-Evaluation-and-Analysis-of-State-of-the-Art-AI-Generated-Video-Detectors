import pandas as pd
import numpy as np

pred_csv = r"D:\HaitongNi\MEngProject\repos\NSG-VD\results\test\default\nsgvd_aigvdbench_predictions.csv"
df = pd.read_csv(pred_csv)
fakes = df[df['label'] == 1].reset_index(drop=True)

# 20个生成器名称 (按照你跑测试时的实际加载顺序)
generators = [
    "AccVideo", "AnimateDiff", "Cogvideox1.5", "EasyAnimate", "HunyuanVideo",
    "IPOC", "LTX", "Open-Sora", "Pyramid-Flow", "RepVideo",
    "SEINE", "SVD", "VideoCrafter", "Wan2.1", "Cogvideox1.5_V2V", 
    "EasyAnimate_I2V", "LTX_I2V", "Pyramid-Flow_I2V", "SEINE_I2V", "SVD_I2V"
]

# 按每25个一组进行切分
results = []
for i, gen_name in enumerate(generators):
    start_idx = i * 25
    end_idx = (i + 1) * 25
    subset = fakes.iloc[start_idx:end_idx]
    if not subset.empty:
        results.append({
            "Generator": gen_name,
            "Recall": subset['pred_label_at_1'].mean()
        })

summary = pd.DataFrame(results)
print("\n=== Final Recall by Generator (Manual Slicing) ===")
print(summary.sort_values(by='Recall', ascending=False).to_string(index=False))