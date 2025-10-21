from pathlib import Path

import numpy as np
from skimage import data, io

# Load image
img = data.camera()
H, W = img.shape

# Parameters for light overlap (horizontal)
overlap_ratio = 0.15  # 15% overlap of the full width
crop_w_ratio = 0.55  # each crop is 55% of the full width

crop_w = int(crop_w_ratio * W)
overlap_px = int(overlap_ratio * W)

# Ensure geometry is valid: start2 + crop_w <= W
start1_x = 0
start2_x = max(0, crop_w - overlap_px)
assert (
    start2_x + crop_w <= W
), "Crop parameters invalid; increase overlap_ratio or reduce crop_w_ratio."

# Full-height horizontal crops with ~15% width overlap
crop1 = img[:, start1_x : start1_x + crop_w]
crop2 = img[:, start2_x : start2_x + crop_w]

# Save
outdir = Path("./examples/inputs2")
outdir.mkdir(parents=True, exist_ok=True)
p0 = outdir / "0.png"
p1 = outdir / "1.png"
io.imsave(str(p0), crop1.astype(np.uint8))
io.imsave(str(p1), crop2.astype(np.uint8))

(
    str(p0),
    str(p1),
    {"H": H, "W": W, "crop_w": crop_w, "overlap_px": overlap_px, "start2_x": start2_x},
)
