#!/usr/bin/env python3
"""Cover-crop simulation: scale the preview to the 320.7x160 art-box aspect
with the CSS cover rule (scale = max(box_w/w, box_h/h), center-crop) and
verify nothing legible is lost — the padded wings should make the crop a
no-op."""
import sys
from pathlib import Path

from PIL import Image

src = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/Users/olivialiau/portfolio-site/assets/pd-drawings-method-schematic.png")
img = Image.open(src)
w, h = img.size
box_w, box_h = 320.7, 160.0
scale = max(box_w / w, box_h / h)
new_w, new_h = round(w * scale), round(h * scale)
img2 = img.resize((new_w, new_h), Image.LANCZOS)
crop_x = round((new_w - box_w) / 2)
crop_y = round((new_h - box_h) / 2)
crop = img2.crop((crop_x, crop_y, crop_x + round(box_w), crop_y + round(box_h)))
print(f"source {w}x{h} -> scaled {new_w}x{new_h} -> cropped {crop.size}")
print(f"pixels cropped: left/right {crop_x}px each, top/bottom {crop_y}px each")
out = Path("/tmp/pd_cover_sim.png")
crop.save(out)
print(f"simulation written: {out}")
if crop_x == 0 and crop_y == 0:
    print("COVER CROP IS A NO-OP — nothing clipped")
else:
    print("WARNING: content clipped — check the simulation")
