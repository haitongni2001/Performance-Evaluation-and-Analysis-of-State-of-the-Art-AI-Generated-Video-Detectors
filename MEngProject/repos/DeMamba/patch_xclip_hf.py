from pathlib import Path

files = [
    Path("models/DeMamba.py"),
    Path("models/XCLIP.py"),
]

old = 'GenVideo/pretrained_weights/xclip'
new = 'microsoft/xclip-base-patch16-zero-shot'

for p in files:
    s = p.read_text(encoding="utf-8")
    if old in s:
        s = s.replace(old, new)
        p.write_text(s, encoding="utf-8")
        print("patched:", p)
    else:
        print("not found:", p)

print("done")