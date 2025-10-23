from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np


def multi_phase_correlation(
    images: List[np.ndarray],
    *,
    tiles: int = 1,
    upsample_factor: int = 10,
    max_shift: Optional[Tuple[int, int]] = None,
    neighbor_span: int = 1,
    min_variance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Non-iterative global phase correlation for >2 images (translation only).

    Args:
        images: list of 2D grayscale arrays, any sizes, float or uint types.
        tiles: run tiled phase correlation (rows*cols == tiles); 1 disables tiling.
        upsample_factor: skimage phase_cross_correlation upsample factor.
        max_shift: optional (max_dy, max_dx) bound per edge for wrap candidates.
        neighbor_span: build edges between i and i+k for k=1..neighbor_span.
        min_variance: skip tiles with variance below this threshold.

    Returns:
        dys: (N,) float32, global dy per image (t_y, with image 0 fixed to 0).
        dxs: (N,) float32, global dx per image (t_x, with image 0 fixed to 0).
        edges_info: list of dicts per measured edge:
            {"i","j","dy","dx","score","H","W"}
            (dy,dx) are the selected pairwise shifts (j relative to i),
            score is NCC used as weight.

    Notes:
        - Handles different image sizes (pads before FFT; NCC on padded overlap).
        - Uses wrap disambiguation via candidate offsets and NCC scoring.
        - Solves a single weighted least-squares system for all translations.
        - No blending/fusion here—only shift estimation.
    """
    from skimage.registration import phase_cross_correlation  # local import

    def _to_f32(I: np.ndarray) -> np.ndarray:
        I = np.asarray(I)
        if I.dtype != np.float32:
            I = I.astype(np.float32)
            # normalize if looks like 8/16-bit
            m = I.max()
            if m > 1.5:
                I /= max(255.0, m)
        return I

    def _pad_to_same(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ha, Wa = A.shape
        Hb, Wb = B.shape
        H, W = max(Ha, Hb), max(Wa, Wb)
        Ap = np.zeros((H, W), np.float32)
        Bp = np.zeros((H, W), np.float32)
        Ap[:Ha, :Wa] = A
        Bp[:Hb, :Wb] = B
        return Ap, Bp

    def _factor_close_to_square(k: int) -> Tuple[int, int]:
        r = int(np.floor(np.sqrt(k)))
        while r > 1 and k % r != 0:
            r -= 1
        return r, k // r

    def _grid_edges(n: int, parts: int) -> np.ndarray:
        return np.linspace(0, n, parts + 1, dtype=int)

    def _run_phase_corr_anysize(
        A: np.ndarray, B: np.ndarray
    ) -> Tuple[float, float, float]:
        Ap, Bp = _pad_to_same(A, B)
        H, W = Ap.shape
        rows, cols = _factor_close_to_square(int(tiles))
        y_edges = _grid_edges(H, rows)
        x_edges = _grid_edges(W, cols)

        best = None
        for i in range(rows):
            for j in range(cols):
                y0, y1 = y_edges[i], y_edges[i + 1]
                x0, x1 = x_edges[j], x_edges[j + 1]
                a = Ap[y0:y1, x0:x1]
                b = Bp[y0:y1, x0:x1]
                if a.size == 0 or b.size == 0:
                    continue
                if a.var() < min_variance or b.var() < min_variance:
                    continue
                shift, error, _ = phase_cross_correlation(
                    a, b, upsample_factor=upsample_factor
                )
                dy, dx = float(shift[0]), float(shift[1])  # rows, cols
                rec = {"dx": dx, "dy": dy, "error": float(error)}
                if best is None or rec["error"] < best["error"]:
                    best = rec
        if best is None:
            # fall back to whole-image (even if flat), to avoid hard failure
            Ap, Bp = _pad_to_same(A, B)
            shift, error, _ = phase_cross_correlation(
                Ap, Bp, upsample_factor=upsample_factor
            )
            dy, dx = float(shift[0]), float(shift[1])
            best = {"dx": dx, "dy": dy, "error": float(error)}
        return best["dx"], best["dy"], best["error"]

    def _candidate_wraps(dy0: int, dx0: int, H: int, W: int) -> List[Tuple[int, int]]:
        Ks = (-1, 0, 1)
        out, seen = [], set()
        for ky in Ks:
            for kx in Ks:
                dy = dy0 + ky * H
                dx = dx0 + kx * W
                if max_shift is not None:
                    my, mx = max_shift
                    if abs(dy) > my or abs(dx) > mx:
                        continue
                if (dy, dx) not in seen:
                    out.append((dy, dx))
                    seen.add((dy, dx))
        return out or [(dy0, dx0)]

    def _ncc_at_shift(A: np.ndarray, B: np.ndarray, dy: int, dx: int) -> float:
        H, W = A.shape
        y0a, y1a = max(0, dy), min(H, H + dy)
        x0a, x1a = max(0, dx), min(W, W + dx)
        y0b, y1b = max(0, -dy), min(H, H - dy)
        x0b, x1b = max(0, -dx), min(W, W - dx)
        if (y1a - y0a) <= 8 or (x1a - x0a) <= 8:
            return -np.inf
        Pa = A[y0a:y1a, x0a:x1a].astype(np.float32)
        Pb = B[y0b:y1b, x0b:x1b].astype(np.float32)
        Pa -= Pa.mean()
        Pb -= Pb.mean()
        denom = (np.linalg.norm(Pa) * np.linalg.norm(Pb)) + 1e-12
        return float((Pa * Pb).sum() / denom)

    def _measure_edge(
        A: np.ndarray, B: np.ndarray
    ) -> Tuple[float, float, float, int, int]:
        # phase corr on padded (tiled), then wrap disambiguation via NCC on padded copies
        dx0, dy0, _ = _run_phase_corr_anysize(A, B)
        Ap, Bp = _pad_to_same(A, B)
        H, W = Ap.shape
        cands = _candidate_wraps(int(round(dy0)), int(round(dx0)), H, W)
        best = (0, 0, -np.inf)
        for dy, dx in cands:
            sc = _ncc_at_shift(Ap, Bp, dy, dx)
            if sc > best[2]:
                best = (dy, dx, sc)
        return best[0], best[1], best[2], H, W  # dy, dx, score, padded size

    # ----- sanitize inputs -----
    ims = [_to_f32(I) for I in images]
    n = len(ims)
    if n < 2:
        raise ValueError("Need at least 2 images.")

    # ----- build simple chain edges by index (i -> i+k) -----
    edges = []
    for i in range(n):
        for k in range(1, neighbor_span + 1):
            j = i + k
            if j < n:
                edges.append((i, j))

    # ----- measure all edges once -----
    edges_info: List[Dict] = []
    for i, j in edges:
        dy, dx, score, H, W = _measure_edge(ims[i], ims[j])
        edges_info.append(
            {"i": i, "j": j, "dy": dy, "dx": dx, "score": score, "H": H, "W": W}
        )

    # ----- weighted least squares (dense normal equations) -----
    # Solve t_j - t_i = d_ij (x and y separately), with t_0 = 0 gauge.
    # Weight by sqrt(max(score,0)+eps).
    eps = 1e-6
    # Build A and b for both axes
    m = len(edges_info) + 1  # +1 for gauge
    A = np.zeros((m, n), dtype=np.float64)
    bx = np.zeros((m,), dtype=np.float64)
    by = np.zeros((m,), dtype=np.float64)

    r = 0
    for e in edges_info:
        w = np.sqrt(max(e["score"], 0.0) + eps)
        i, j = e["i"], e["j"]
        A[r, i] = -w
        A[r, j] = w
        bx[r] = w * e["dx"]
        by[r] = w * e["dy"]
        r += 1

    # gauge constraint: t_0 = 0 with strong weight
    A[r, 0] = 1e3
    bx[r] = 0.0
    by[r] = 0.0

    # Normal equations (A^T A) t = A^T b
    ATA = A.T @ A
    ATbx = A.T @ bx
    ATby = A.T @ by

    # Regularize slightly to avoid singularity if graph is weak
    reg = 1e-8 * np.eye(n, dtype=np.float64)
    tx = np.linalg.solve(ATA + reg, ATbx)
    ty = np.linalg.solve(ATA + reg, ATby)

    dxs = tx.astype(np.float32)
    dys = ty.astype(np.float32)
    return dxs, dys, edges_info
