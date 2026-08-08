import os

target_file = r"D:\HaitongNi\MEngProject\repos\NSG-VD\test_dMMD.py"

with open(target_file, "r", encoding="utf-8") as f:
    code = f.read()

if "nsgvd_aigvdbench" in code:
    print("✅ 已经升级过了，可以直接跑 Inference！")
    exit()

if "pred_csv_path" not in code:
    print("❌ 警告：你的 test_dMMD.py 目前是未打补丁的原始状态。")
    print("👉 请先运行: python patch_nsgvd_save_predictions.py")
    print("👉 然后再运行一次本脚本。")
    exit()

# 1. 动态化 CSV 文件名
code = code.replace(
    'pred_csv_path = os.path.join(cfg.log_path, "nsgvd_genvidbench_common1000_predictions.csv")',
    'pred_csv_path = os.path.join(cfg.log_path, f"nsgvd_aigvdbench_predictions.csv")'
)

# 2. 改为 Append 追加模式 (防止循环测试时数据被互相覆盖)
old_save = 'pd.DataFrame(pred_rows).to_csv(pred_csv_path, index=False, encoding="utf-8-sig")'
new_save = '''import os
            file_exists = os.path.isfile(pred_csv_path)
            pd.DataFrame(pred_rows).to_csv(pred_csv_path, mode='a', header=not file_exists, index=False, encoding="utf-8-sig")'''
code = code.replace(old_save, new_save)

# 3. 强化 AIGVDBench 的生成器名字提取逻辑 (兼容文件夹路径分割)
old_gen = 'gen = "_".join(parts[2:]) if len(parts) >= 3 else name'
new_gen = 'gen = sample_id.replace("\\\\", "/").split("/")[0] if "/" in sample_id.replace("\\\\", "/") else name'
code = code.replace(old_gen, new_gen)

with open(target_file, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ SUCCESS! test_dMMD.py 代码已完美升级，完全兼容 AIGVDBench！")