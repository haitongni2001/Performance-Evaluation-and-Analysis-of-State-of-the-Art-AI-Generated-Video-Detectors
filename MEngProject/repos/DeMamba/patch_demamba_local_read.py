from pathlib import Path

p = Path("dataloader.py")
s = p.read_text(encoding="utf-8")

# Make sure cv2 is imported.
if "import cv2" not in s:
    s = s.replace("import random\n", "import random\nimport cv2\n")

# Replace remote/object-storage style reading with local image reading.
s = s.replace(
    "image = download_oss_file('GenVideo/'+ temp_image_path)",
    "image = cv2.imread('GenVideo/' + temp_image_path)\n                        if image is None:\n                            raise FileNotFoundError('GenVideo/' + temp_image_path)"
)

s = s.replace(
    "image = download_oss_file('GenVideo/'+temp_image_path)",
    "image = cv2.imread('GenVideo/' + temp_image_path)\n                if image is None:\n                    raise FileNotFoundError('GenVideo/' + temp_image_path)"
)

s = s.replace(
    "image = download_oss_file('GenVideo/'+ temp_image_path)",
    "image = cv2.imread('GenVideo/' + temp_image_path)\n                        if image is None:\n                            raise FileNotFoundError('GenVideo/' + temp_image_path)"
)

p.write_text(s, encoding="utf-8")
print("patched dataloader.py to read local jpg files with cv2.imread")