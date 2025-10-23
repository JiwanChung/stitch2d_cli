from __future__ import annotations

from pathlib import Path
import base64
import io
import traceback
import numpy as np
import cv2

import flet as ft
from skimage.io import imsave

# --- import your existing core functions ---
from stitch2d_cli.function import fft_phase_correlation
from stitch2d_cli.fuse import fuse_two_images


# ---------------- utilities ----------------
def read_gray_f32(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img.astype(np.float32) / 255.0


def match_global_mean_var(A: np.ndarray, B: np.ndarray):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    a = (A.std() + 1e-12) / (B.std() + 1e-12)
    b = A.mean() - a * B.mean()
    Bp = a * B + b
    return Bp, float(a), float(b)


def ndarray_to_base64_png(img: np.ndarray) -> str:
    """Return a base64-encoded PNG data URL for displaying in Flet Image."""
    if img.dtype != np.uint8:
        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img8 = img
    # Ensure 2D grayscale becomes 3-channel for consistent display
    if img8.ndim == 2:
        img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", img8)
    if not ok:
        raise RuntimeError("Failed to encode preview PNG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def stitch_pair(file1: Path, file2: Path):
    I1, I2 = read_gray_f32(file1), read_gray_f32(file2)
    I2_adj, a, b = match_global_mean_var(I1, I2)
    dx, dy, resp = fft_phase_correlation(I1, I2_adj)
    fused = fuse_two_images(I1, I2_adj, (dx, dy))
    meta = {"dx": dx, "dy": dy, "resp": resp, "gain": a, "offset": b}
    return fused, meta


def iter_files(paths: list[Path]) -> np.ndarray:
    paths = list(sorted(paths, key=lambda p: p.stem))
    if len(paths) < 2:
        raise ValueError("Need at least two images to stitch.")
    fused = read_gray_f32(paths[0])
    for p in paths[1:]:
        nxt = read_gray_f32(p)
        nxt_adj, _, _ = match_global_mean_var(fused, nxt)
        dx, dy, _ = fft_phase_correlation(fused, nxt_adj)
        fused = fuse_two_images(fused, nxt_adj, (dx, dy))
    return fused


def run_batch(root: Path, log_cb=lambda *_: None) -> int:
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(root.glob("*")):
        if not d.is_dir() or d.name == "outputs":
            continue
        inputs = [
            *d.glob("*.png"),
            *d.glob("*.jpg"),
            *d.glob("*.jpeg"),
            *d.glob("*.tif"),
            *d.glob("*.tiff"),
        ]
        if len(inputs) < 2:
            continue
        try:
            fused = iter_files(inputs)
            out_path = out_dir / f"{d.name}.png"
            imsave(str(out_path), (np.clip(fused, 0, 1) * 255).astype(np.uint8))
            n += 1
            log_cb(f"✓ saved fused image → {out_path}")
        except Exception as e:
            log_cb(f"✗ error in {d}: {e}")
    return n


# ---------------- Flet app ----------------
def main(page: ft.Page):
    page.title = "Stitch2D GUI (phase correlation + NCC-style refinement)"
    page.window_width = 1100
    page.window_height = 800
    page.theme_mode = ft.ThemeMode.SYSTEM

    # ---------- shared helpers ----------
    def append_log(control: ft.TextField, msg: str):
        control.value = (control.value + msg + "\n") if control.value else (msg + "\n")
        control.update()

    # ---------- Pair tab ----------
    f1_path = ft.TextField(label="Reference (left)", expand=True)
    f2_path = ft.TextField(label="Moving (right)", expand=True)
    out_path = ft.TextField(label="Output (PNG)", value="fused.png", expand=True)

    pair_log = ft.TextField(
        label="Log",
        multiline=True,
        min_lines=6,
        max_lines=10,
        read_only=True,
        expand=True,
    )
    preview = ft.Image(
        src=None,
        width=960,
        height=540,
        fit=ft.ImageFit.CONTAIN,
        border_radius=8,
    )

    # File pickers
    fp1 = ft.FilePicker(
        on_result=lambda e: (
            (setattr(f1_path, "value", e.files[0].path), f1_path.update())
            if e.files
            else None
        )
    )
    fp2 = ft.FilePicker(
        on_result=lambda e: (
            (setattr(f2_path, "value", e.files[0].path), f2_path.update())
            if e.files
            else None
        )
    )
    fp_out = ft.FilePicker(
        on_result=lambda e: (
            (setattr(out_path, "value", e.path), out_path.update()) if e.path else None
        )
    )
    page.overlay.extend([fp1, fp2, fp_out])

    def on_pair_run(_):
        try:
            p1, p2 = Path(f1_path.value), Path(f2_path.value)
            if not (p1 and p1.exists() and p2 and p2.exists()):
                append_log(pair_log, "Select two valid image files.")
                return
            fused, meta = stitch_pair(p1, p2)
            append_log(
                pair_log,
                f"[PhaseCorr] dx={meta['dx']:.2f}, dy={meta['dy']:.2f}, resp={meta['resp']:.3f} | gain={meta['gain']:.4f}, offset={meta['offset']:.4f}",
            )
            preview.src = ndarray_to_base64_png(fused)
            preview.update()
            outp = Path(out_path.value) if out_path.value else Path("fused.png")
            outp.parent.mkdir(parents=True, exist_ok=True)
            imsave(str(outp), (np.clip(fused, 0, 1) * 255).astype(np.uint8))
            append_log(pair_log, f"✅ Saved fused image → {outp}")
        except Exception as e:
            tb = "".join(traceback.format_exception(e))
            append_log(pair_log, tb)

    def on_pair_clear(_):
        preview.src = None
        preview.update()
        pair_log.value = ""
        pair_log.update()

    pair_controls = ft.Column(
        [
            ft.Row(
                [
                    f1_path,
                    ft.ElevatedButton(
                        "Browse",
                        icon=ft.Icons.FOLDER_OPEN,
                        on_click=lambda _: fp1.pick_files(
                            allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE
                        ),
                    ),
                ]
            ),
            ft.Row(
                [
                    f2_path,
                    ft.ElevatedButton(
                        "Browse",
                        icon=ft.Icons.FOLDER_OPEN,
                        on_click=lambda _: fp2.pick_files(
                            allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE
                        ),
                    ),
                ]
            ),
            ft.Row(
                [
                    out_path,
                    ft.ElevatedButton(
                        "Save as…",
                        icon=ft.Icons.SAVE,
                        on_click=lambda _: fp_out.save_file(
                            file_type=ft.FilePickerFileType.CUSTOM,
                            allowed_extensions=["png"],
                        ),
                    ),
                ]
            ),
            ft.Row(
                [
                    ft.ElevatedButton(
                        "Preview & Stitch",
                        icon=ft.Icons.PLAY_ARROW,
                        on_click=on_pair_run,
                    ),
                    ft.TextButton("Clear", on_click=on_pair_clear),
                ]
            ),
            preview,
            pair_log,
        ],
        spacing=12,
        expand=True,
    )

    # ---------- Batch tab ----------
    root_dir = ft.TextField(
        label="Root folder (contains multiple subfolders with images)", expand=True
    )
    batch_log = ft.TextField(
        label="Batch log",
        multiline=True,
        min_lines=12,
        max_lines=20,
        read_only=True,
        expand=True,
    )
    folder_picker = ft.FilePicker(
        on_result=lambda e: (
            (setattr(root_dir, "value", e.path), root_dir.update()) if e.path else None
        )
    )
    page.overlay.append(folder_picker)

    progress = ft.ProgressBar(width=300, visible=False)

    def open_outputs(_):
        if not root_dir.value:
            append_log(batch_log, "Choose a root folder first.")
            return
        outd = Path(root_dir.value) / "outputs"
        outd.mkdir(parents=True, exist_ok=True)
        page.launch_url(outd.as_uri())

    def batch_worker():
        try:
            r = Path(root_dir.value)
            if not r.exists():
                append_log(batch_log, f"Invalid folder: {r}")
                return

            def logger(msg: str):
                append_log(batch_log, msg)

            n = run_batch(r, log_cb=logger)
            append_log(batch_log, f"Completed directory {r}: {n} images saved")
        finally:
            progress.visible = False
            progress.update()

    def on_batch_run(_):
        if not root_dir.value:
            append_log(batch_log, "Choose a root folder.")
            return
        batch_log.value = ""
        batch_log.update()
        progress.visible = True
        progress.update()
        # Run without freezing UI
        page.run_task(batch_worker)

    batch_controls = ft.Column(
        [
            ft.Row(
                [
                    root_dir,
                    ft.ElevatedButton(
                        "Browse",
                        icon=ft.Icons.FOLDER_OPEN,
                        on_click=lambda _: folder_picker.get_directory_path(),
                    ),
                ]
            ),
            ft.Row(
                [
                    ft.ElevatedButton(
                        "Run Batch", icon=ft.Icons.PLAYLIST_PLAY, on_click=on_batch_run
                    ),
                    ft.TextButton("Open outputs", on_click=open_outputs),
                    progress,
                ]
            ),
            batch_log,
        ],
        spacing=12,
        expand=True,
    )

    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="Pair", content=pair_controls, icon=ft.Icons.JOIN_LEFT),
            ft.Tab(text="Batch", content=batch_controls, icon=ft.Icons.FOLDER_COPY),
        ],
        expand=True,
    )

    page.add(tabs)


if __name__ == "__main__":
    # Desktop app
    ft.app(target=main, view=ft.AppView.FLET_APP)
