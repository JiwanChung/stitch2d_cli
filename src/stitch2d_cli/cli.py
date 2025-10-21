import typer
from pathlib import Path

import numpy as np
import cv2
from skimage.io import imsave

from stitch2d_cli.function import fft_phase_correlation
from stitch2d_cli.fuse import fuse_two_images


app = typer.Typer(
    help="Robust 2D stitching (ImageJ-style phase correlation + NCC refinement)"
)


# ---------------- Utility functions ----------------
def read_gray_f32(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def match_global_mean_var(A: np.ndarray, B: np.ndarray):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    a = (A.std() + 1e-12) / (B.std() + 1e-12)
    b = A.mean() - a * B.mean()
    Bp = a * B + b
    return Bp, float(a), float(b)


# ---------------- Main CLI command ----------------
@app.command()
def pair(
    file1: Path = typer.Argument(..., help="Reference (left) image"),
    file2: Path = typer.Argument(..., help="Moving (right) image"),
    output: Path = typer.Option(
        "fused.png", "--output", "-o", help="Output fused image"
    ),
    blend_width: int = typer.Option(
        60, "--blend-width", "-b", help="Feather blending width in px"
    ),
):
    """Perform ImageJ-like 2D stitching with bandpass and NCC refinement."""
    I1, I2 = read_gray_f32(file1), read_gray_f32(file2)
    I2, _, _ = match_global_mean_var(I1, I2)
    print(I1.mean())
    print(I2.mean())

    dx, dy, resp = fft_phase_correlation(I1, I2)
    typer.echo(f"[PhaseCorr] dx={dx:.2f}, dy={dy:.2f}, resp={resp:.3f}")

    fused = fuse_two_images(I1, I2, (dx, dy))

    output.parent.mkdir(parents=True, exist_ok=True)
    imsave(str(output), (np.clip(fused, 0, 1) * 255).astype(np.uint8))
    typer.echo(f"✅ Saved fused image → {output}")


def main():
    app()


if __name__ == "__main__":
    main()
