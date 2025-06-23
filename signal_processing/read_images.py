import numpy as np
from pathlib import Path
import imageio.v2 as imageio     # pip install imageio

NPY_FILE = Path(r"C:\Users\admin\Desktop\phd-workspace\AcquireData\dataset\u511c\2025-06-21_19-58-54.296401\frameblock_00000.npy")
OUT_DIR  = Path(r"C:\Users\admin\Desktop\phd-workspace\AcquireData\dataset\u511c\2025-06-21_19-58-54.296401\png")

FLIP_BGR = False

OUT_DIR.mkdir(parents=True, exist_ok=True)

frames = np.load(NPY_FILE)        # shape (N, H, W)  lub  (N, H, W, C)
n_frames = frames.shape[0]

# pierwsze 10 + co 10-ta
idxs = list(range(min(10, n_frames))) + list(range(10, n_frames, 10))

for idx in idxs:
    img = frames[idx]

    if img.ndim == 3 and img.shape[2] == 3:          # 3-kanalowe
        if FLIP_BGR:
            img = img[..., ::-1]                     # BGR->RGB
    elif img.ndim == 3 and img.shape[2] == 1:        # (H, W, 1) -> (H, W)
        img = img.squeeze(-1)

    fname = OUT_DIR / f"frame_{idx:05d}.png"
    imageio.imwrite(fname, img)
    print(f"saved {fname}")

print(f"Done - wrote {len(idxs)} PNGs to {OUT_DIR}")
