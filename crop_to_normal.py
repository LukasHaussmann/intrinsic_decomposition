"""Crop rgb_*.png images to match normal.png resolution, centred vertically."""
import os
from PIL import Image

DATA_DIR = "data/blender"

normal = Image.open(os.path.join(DATA_DIR, "normal.png"))
target_w, target_h = normal.size

rgb_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("rgb") and f.endswith(".png")
)

for fname in rgb_files:
    path = os.path.join(DATA_DIR, fname)
    img = Image.open(path)
    w, h = img.size

    # centre-crop: trim equal amounts from top and bottom
    top  = (h - target_h) // 2
    left = (w - target_w) // 2
    img_cropped = img.crop((left, top, left + target_w, top + target_h))

    img_cropped.save(path)
    print(f"{fname}: {w}×{h} → {img_cropped.width}×{img_cropped.height}")
