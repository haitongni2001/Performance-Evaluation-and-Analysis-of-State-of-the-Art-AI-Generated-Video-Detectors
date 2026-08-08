from pathlib import Path

p = Path("util.py")
s = p.read_text(encoding="utf-8")

replacements = {
    "FN = int((1-accuracy) * video_nums[index])":
        "FN = int((1-accuracy) * video_nums[index]) if not np.isnan(accuracy) else 0",

    "TP = int(accuracy * video_nums[index])":
        "TP = int(accuracy * video_nums[index]) if not np.isnan(accuracy) else 0",

    "FP = int((1-accuracy) * video_nums[index])":
        "FP = int((1-accuracy) * video_nums[index]) if not np.isnan(accuracy) else 0",

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