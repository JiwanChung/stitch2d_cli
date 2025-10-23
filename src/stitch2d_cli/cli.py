import os
from pathlib import Path
from collections import defaultdict

import typer
import numpy as np
import cv2
from tqdm import tqdm
from skimage.io import imsave

from stitch2d_cli.pair import pair_phase_correlation
from stitch2d_cli.multi import multi_phase_correlation
from stitch2d_cli.fuse import fuse_two_images
from stitch2d_cli.fuse_multi import fuse_many_images
from stitch2d_cli.utils import match_global_mean_var_multi


app = typer.Typer(
    help="Robust 2D stitching (ImageJ-style phase correlation + NCC refinement)"
)


# ---------------- Utility functions ----------------
def read_gray_f32(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


# def match_global_mean_var(A: np.ndarray, B: np.ndarray):
#     A = A.astype(np.float32)
#     B = B.astype(np.float32)
#     a = (A.std() + 1e-12) / (B.std() + 1e-12)
#     b = A.mean() - a * B.mean()
#     Bp = a * B + b
#     return Bp, float(a), float(b)


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
    (I1, I2), _ = match_global_mean_var_multi([I1, I2])

    dx, dy, resp = pair_phase_correlation(I1, I2)
    typer.echo(f"[PhaseCorr] dx={dx:.2f}, dy={dy:.2f}, resp={resp:.3f}")

    fused = fuse_two_images(I1, I2, (dx, dy))

    output.parent.mkdir(parents=True, exist_ok=True)
    imsave(str(output), (np.clip(fused, 0, 1) * 255).astype(np.uint8))
    typer.echo(f"✅ Saved fused image → {output}")


def do_stitch_pair(I1, I2):
    (I1, I2), _ = match_global_mean_var_multi([I1, I2])

    dx, dy, resp = pair_phase_correlation(I1, I2)
    typer.echo(f"[PhaseCorr] dx={dx:.2f}, dy={dy:.2f}, resp={resp:.3f}")

    fused = fuse_two_images(I1, I2, (dx, dy))
    return fused


def do_stitch_multi(*images):
    images, _ = match_global_mean_var_multi(images)

    dxs, dys, _ = multi_phase_correlation(images)

    fused = fuse_many_images(images, list(zip(dxs, dys)))
    return fused


def iter_files(paths: list[Path], do_sort: bool = False) -> np.ndarray:
    if do_sort:
        paths = list(sorted(paths, key=lambda x: x.stem))
    head, rest = paths[0], paths[1:]
    curr_image = read_gray_f32(head)
    for next_path in rest:
        next_image = read_gray_f32(next_path)
        fused = do_stitch_pair(curr_image, next_image)
        curr_image = fused
    return fused


def _clean_dir_arg(s: str) -> Path:
    # Strip surrounding quotes users often add on Windows
    s = s.strip().strip('"').strip("'")
    # Norm and expand (~), keep non-existent paths valid
    p = Path(os.path.normpath(os.path.expanduser(s)))
    # Guard against wildcards or illegal characters on Windows
    bad = set('*?"<>|')
    if any(c in str(p) for c in bad):
        raise typer.BadParameter("Input path must be a directory, not a glob/pattern.")
    return p


@app.command()
def batch(path: str):
    root = _clean_dir_arg(path)
    if not root.exists() or not root.is_dir():
        raise typer.BadParameter(f"Directory not found: {root}")
    out_dir = Path(root) / "outputs"
    out_dir.mkdir(exist_ok=True, parents=True)
    n = 0
    for dir in Path(root).glob("*"):
        if not dir.is_dir():
            continue
        if dir.name == "outputs":
            continue

        inputs = [*dir.glob("*.png"), *dir.glob("*.jpg")]
        if len(inputs) < 2:
            continue
        fused = iter_files(inputs, do_sort=True)
        out_path = out_dir / (dir.name + ".png")

        imsave(str(out_path), (np.clip(fused, 0, 1) * 255).astype(np.uint8))
        n += 1
        typer.echo(f"saved fused image -> {out_path}")

    typer.echo(f"completed directory {path}: {n} images saved")


def sort_files(li: list[tuple[int, Path]]) -> list[tuple[int, Path]]:
    return list(sorted(li, key=lambda x: x[0]))


@app.command()
def recurse(path: str):
    root = _clean_dir_arg(path)
    if not root.exists() or not root.is_dir():
        raise typer.BadParameter(f"Directory not found: {root}")
    out_dir = Path(root) / "outputs"
    out_dir.mkdir(exist_ok=True, parents=True)

    jobs = []

    def _register_job(dir: Path):
        if dir.name == "outputs":
            return
        inputs = [*dir.glob("*.png"), *dir.glob("*.jpg")]
        if len(inputs) < 1:
            return
        # group by names

        groups = defaultdict(lambda: [])
        for input in inputs:
            name = input.stem
            if len(name) > 1:
                key, val = name[:-1], int(name[-1])
                groups[key].append([val, input])

        groups = {k: v for k, v in groups.items() if len(v) > 1}
        groups = {k: sort_files(v) for k, v in groups.items()}
        for k, inputs in groups.items():
            imgs = [v[1] for v in inputs]
            jobs.append([dir, k, imgs])

    def register_job(dir: Path):
        _register_job(dir)
        for subdir in dir.glob("*"):
            if subdir.is_dir():
                register_job(subdir)

    # recurse the root
    register_job(root)

    n = 0
    for dir, key, img_paths in tqdm(jobs, desc="running stitch jobs"):
        name = dir.relative_to(root).as_posix()  # normalize separators to "/"
        name = name.replace(" ", "_").replace("/", "__")
        name = f"{name}_{key}.png"
        out_path = out_dir / name
        print(out_path)
        images = [read_gray_f32(p) for p in img_paths]
        fused = do_stitch_multi(*images)
        imsave(str(out_path), (np.clip(fused, 0, 1) * 255).astype(np.uint8))
        n += 1
        tqdm.write(f"saved fused image -> {out_path}")
    typer.echo(f"completed directory {root}: {n} images saved")


def main():
    app()


if __name__ == "__main__":
    main()
