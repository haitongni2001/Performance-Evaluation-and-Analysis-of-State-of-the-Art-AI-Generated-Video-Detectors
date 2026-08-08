from pathlib import Path

p = Path("util.py")
s = p.read_text(encoding="utf-8")

# Avoid crash when a subgroup accuracy is NaN.
# Original line:
# FP = int((1-accuracy) * 10000)
s = s.replace(
    "FP = int((1-accuracy) * 10000)",
    "FP = int((1-accuracy) * 10000) if not np.isnan(accuracy) else 0"
)

# Also patch possible similar patterns if they exist.
s = s.replace(
    "FN = int((1-recall) * 10000)",
    "FN = int((1-recall) * 10000) if not np.isnan(recall) else 0"
)

p.write_text(s, encoding="utf-8")
print("patched util.py NaN metric crash")