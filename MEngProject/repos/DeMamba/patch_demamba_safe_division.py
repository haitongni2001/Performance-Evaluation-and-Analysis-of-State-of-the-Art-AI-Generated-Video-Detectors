from pathlib import Path

p = Path("util.py")
s = p.read_text(encoding="utf-8")

replacements = {
    # GenVideo subgroup precision/recall
    "P, R = TP / (TP + FP), TP / (TP + FN)":
        "P = TP / (TP + FP) if (TP + FP) > 0 else 0\n            R = TP / (TP + FN) if (TP + FN) > 0 else 0",

    # possible F1 calculation
    "F1 = 2 * P * R / (P + R)":
        "F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0",

    # NaN guards from earlier
    "TP = int(accuracy * video_nums[index])":
        "TP = int(accuracy * video_nums[index]) if not np.isnan(accuracy) else 0",

    "FP = int((1-accuracy) * video_nums[index])":
        "FP = int((1-accuracy) * video_nums[index]) if not np.isnan(accuracy) else 0",

    "FN = int((1-accuracy) * video_nums[index])":
        "FN = int((1-accuracy) * video_nums[index]) if not np.isnan(accuracy) else 0",

    "FP = int((1-accuracy) * 10000)":
        "FP = int((1-accuracy) * 10000) if not np.isnan(accuracy) else 0",

    "FN = int((1-recall) * 10000)":
        "FN = int((1-recall) * 10000) if not np.isnan(recall) else 0",
}

for old, new in replacements.items():
    if old in s:
        s = s.replace(old, new)
        print("patched:", old)
    else:
        print("not found or already patched:", old)

p.write_text(s, encoding="utf-8")
print("done")