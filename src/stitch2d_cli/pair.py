from typing import Tuple, List, Optional

import numpy as np
from skimage.registration import phase_cross_correlation


def _factor_close_to_square(k: int) -> tuple[int, int]:
    """Return (rows, cols) such that rows*cols == k and the grid is as square as possible."""
    r = int(np.floor(np.sqrt(k)))
    while r > 1 and k % r != 0:
        r -= 1
    return r, k // r


def _grid_edges(n: int, parts: int) -> np.ndarray:
    """Integer edges that partition [0, n] into 'parts' nearly-equal chunks."""
    return np.linspace(0, n, parts + 1, dtype=int)


def run_phase_corr(
    img1: np.ndarray,
    img2: np.ndarray,
    tiles: int = 1,
    upsample_factor: int = 10,
    min_variance: float = 1e-6,
):
    """
    Split both images into 'tiles' pieces (rows*cols == tiles), run phase_cross_correlation
    on each corresponding tile, and return the best result.

    Returns:
        best: (dx, dy, error, box) where box = (y, x, h, w)
        details: list of per-tile dicts: {"box",(dx,dy), "error"}
    """
    A = np.asarray(img1)
    B = np.asarray(img2)
    assert A.ndim == 2 and B.ndim == 2, "Images must be 2D grayscale."
    if A.shape != B.shape:
        raise ValueError("Images must have the same shape.")

    H, W = A.shape
    rows, cols = _factor_close_to_square(int(tiles))
    y_edges = _grid_edges(H, rows)
    x_edges = _grid_edges(W, cols)

    results = []
    for i in range(rows):
        for j in range(cols):
            y0, y1 = y_edges[i], y_edges[i + 1]
            x0, x1 = x_edges[j], x_edges[j + 1]
            h, w = y1 - y0, x1 - x0
            if h == 0 or w == 0:
                continue

            a = A[y0:y1, x0:x1]
            b = B[y0:y1, x0:x1]

            # Skip flat tiles (no texture → unreliable)
            if a.var() < min_variance or b.var() < min_variance:
                continue

            shift, error, diffphase = phase_cross_correlation(
                a, b, upsample_factor=upsample_factor
            )
            dy, dx = float(shift[0]), float(shift[1])  # (rows, cols)
            results.append(
                {
                    "box": (y0, x0, h, w),
                    "dx": dx,
                    "dy": dy,
                    "error": float(error),
                }
            )

    if not results:
        raise RuntimeError("No valid tiles (all were too flat or empty).")

    best = min(results, key=lambda r: r["error"])
    best_tuple = (best["dx"], best["dy"], best["error"], best["box"])
    return best_tuple[:3]


def candidate_wraps(
    dy0: int, dx0: int, H: int, W: int, max_shift: Optional[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Generate dy,dx plus wrapped variants within bounds."""
    Ks = [-1, 0, 1]
    candidates = []
    for ky in Ks:
        for kx in Ks:
            dy = dy0 + ky * H
            dx = dx0 + kx * W
            if max_shift is None:
                candidates.append((dy, dx))
            else:
                my, mx = max_shift
                if abs(dy) <= my and abs(dx) <= mx:
                    candidates.append((dy, dx))
    # de-dup while preserving order
    uniq = []
    seen = set()
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def ncc_at_shift(A: np.ndarray, B: np.ndarray, dy: int, dx: int) -> float:
    """Normalized cross-correlation on the overlapping area for integer (dy,dx)."""
    H, W = A.shape
    y0a = max(0, dy)
    y1a = min(H, H + dy)
    x0a = max(0, dx)
    x1a = min(W, W + dx)
    y0b = max(0, -dy)
    y1b = min(H, H - dy)
    x0b = max(0, -dx)
    x1b = min(W, W - dx)

    if (y1a - y0a) <= 8 or (x1a - x0a) <= 8:
        return -np.inf  # too little overlap to be meaningful

    Pa = A[y0a:y1a, x0a:x1a].astype(np.float32)
    Pb = B[y0b:y1b, x0b:x1b].astype(np.float32)

    Pa = Pa - Pa.mean()
    Pb = Pb - Pb.mean()
    denom = (np.linalg.norm(Pa) * np.linalg.norm(Pb)) + 1e-12
    return float((Pa * Pb).sum() / denom)


def pair_phase_correlation(
    A: np.ndarray,
    B: np.ndarray,
    max_shift: Optional[tuple[int, int]] = None,
) -> tuple[int, int, float]:
    """
    Phase correlation with wrap disambiguation via spatial NCC.
    max_shift: (max_dy, max_dx). If given, we prefer candidates within this bound.
    Returns (dy, dx, ncc_score).
    """

    H, W = A.shape
    dx0, dy0, error = run_phase_corr(A, B)
    dx0 = int(dx0)
    dy0 = int(dy0)

    # candidate set (principal + wrapped neighbors)
    cands = candidate_wraps(dy0, dx0, H, W, max_shift)

    # if no max_shift, also keep principal as fallback
    if not cands:
        cands = [(dy0, dx0)]

    # score in spatial domain
    best = (dy0, dx0, -np.inf)
    for dy, dx in cands:
        score = ncc_at_shift(A, B, dy, dx)
        if score > best[2]:
            best = (dx, dy, score)
    return best
