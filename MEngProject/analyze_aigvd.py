import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

d3_path = r"D:\HaitongNi\MEngProject\repos\D3\results\predictions_20260512_101813.csv"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("=== Analyzing D3 on AIGVDBench (Dynamic Best Threshold) ===")
df = pd.read_csv(d3_path)

# 1. 归一化：将未校准的 Logits 转换为 0-1 之间的概率
df['prob_score'] = df['fake_score'].apply(sigmoid)

y_true = df['label']
y_scores = df['prob_score']

# 2. 计算与阈值无关的绝对排序能力 (AUC & AP)
auc = roc_auc_score(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
print(f"Overall ROC-AUC: {auc:.4f}")
print(f"Overall Average Precision (AP): {ap:.4f}")

# 3. 计算动态最佳阈值 (使用 Youden's J statistic)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
J = tpr - fpr
best_idx = np.argmax(J)
best_threshold = thresholds[best_idx]

print(f"Best Derived Threshold: {best_threshold:.4f}")
print("-" * 50)
print("Per-generator Breakdown (Using Best Threshold):")

# 4. 严格计算召回率 (完美避开旧版 Pandas 警告的写法)
fakes = df[df['label'] == 1]
recall_by_gen = fakes.groupby('generator_id')['prob_score'].apply(
    lambda x: (x >= best_threshold).mean()
).reset_index()

recall_by_gen.columns = ['Generator', 'Recall']
recall_by_gen = recall_by_gen.sort_values(by='Recall')

print(recall_by_gen.to_string(index=False))