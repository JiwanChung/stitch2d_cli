import numpy as np
import cv2 as cv


def _to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _lin_gain_offset(src, ref, mask):
    """
    Fit per-channel a, b in: a*src + b ≈ ref over masked overlap (least squares).
    Returns (a, b) of shape (C,).
    """
    eps = 1e-8
    h, w = mask.shape
    m = mask.reshape(-1) > 0

    src_f = src.reshape(-1, src.shape[2])[m]
    ref_f = ref.reshape(-1, ref.shape[2])[m]
    if src_f.size == 0:
        # No overlap → neutral transform
        C = src.shape[2]
        return np.ones((C,), np.float32), np.zeros((C,), np.float32)

    # Solve per channel: minimize ||a*x + b - y||^2
    X = np.stack([src_f, np.ones_like(src_f)], axis=2)  # [N,C,2]
    # Closed-form solution for each channel
    a = np.empty(src.shape[2], np.float32)
    b = np.empty(src.shape[2], np.float32)
    for c in range(src.shape[2]):
        xc = X[:, c, 0]
        ones = X[:, c, 1]
        yc = ref_f[:, c]
        # normal equations
        Sx = np.sum(xc)
        Sy = np.sum(yc)
        Sxx = np.sum(xc * xc)
        Sxy = np.sum(xc * yc)
        N = xc.shape[0]
        denom = (Sxx * N - Sx * Sx) + eps
        a[c] = (Sxy * N - Sx * Sy) / denom
        b[c] = (Sy - a[c] * Sx) / (N + eps)
    return a, b


def _distance_weight(mask):
    """
    Build soft weights from a binary mask using distance transform.
    mask: uint8 {0,1}, foreground=1.
    """
    # distance to background
    dist_in = cv.distanceTransform((mask * 255).astype(np.uint8), cv.DIST_L2, 3)
    # prevent exact zeros (avoid div by zero later)
    w = dist_in + 1e-3
    return w


def _place_on_canvas(img, offset, canvas_shape):
    """
    Place img on a larger canvas at (dx, dy) offset (x=col, y=row).
    Returns placed image and binary mask.
    """
    Hc, Wc = canvas_shape[:2]
    H, W = img.shape[:2]
    dx, dy = offset  # x to right, y down

    x0, y0 = int(round(max(dx, 0))), int(round(max(dy, 0)))
    x1, y1 = x0 + min(W, Wc - x0), y0 + min(H, Hc - y0)

    sx0 = int(round(max(-dx, 0)))
    sy0 = int(round(max(-dy, 0)))
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    canvas = np.zeros(canvas_shape, np.float32)
    mask = np.zeros((Hc, Wc), np.uint8)

    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = img[sy0:sy1, sx0:sx1]
        mask[y0:y1, x0:x1] = 1
    return canvas, mask


def _compute_canvas_shape(imgA, imgB, offset):
    H1, W1 = imgA.shape[:2]
    H2, W2 = imgB.shape[:2]
    dx, dy = offset
    # bounding box covering imgA at (0,0) and imgB at (dx,dy)
    x_min = min(0, dx)
    y_min = min(0, dy)
    x_max = max(W1, dx + W2)
    y_max = max(H1, dy + H2)
    Wc = x_max - x_min
    Hc = y_max - y_min
    # shift both so that min corner is (0,0)
    shift = (-x_min, -y_min)
    return (Hc, Wc), shift


def fuse_two_images(imgA, imgB, offset_xy, strength=1.0, clamp=True):
    """
    Smoothly fuse imgA and a shifted imgB.
    - offset_xy = (dx, dy) meaning imgB is located dx pixels right, dy pixels down from imgA’s origin.
    - strength in [0..1]: 1.0 = full distance-based feather, lower = more uniform mix in overlap.
    """
    A = _to_float(imgA)
    B = _to_float(imgB)
    if A.ndim == 2:
        A = A[..., None]
    if B.ndim == 2:
        B = B[..., None]
    if A.shape[2] != B.shape[2]:
        raise ValueError("Channel count mismatch.")

    canvas_shape, shift = _compute_canvas_shape(A, B, offset_xy)
    # Place A at shift, B at (shift + offset)
    A_can, Am = _place_on_canvas(A, shift, (*canvas_shape, A.shape[2]))
    B_can, Bm = _place_on_canvas(
        B,
        (shift[0] + offset_xy[0], shift[1] + offset_xy[1]),
        (*canvas_shape, B.shape[2]),
    )

    overlap = (Am & Bm).astype(np.uint8)
    onlyA = (Am & (~Bm.astype(bool))).astype(np.uint8)
    onlyB = ((~Am.astype(bool)) & Bm).astype(np.uint8)

    # Exposure compensation in overlap: fit A to B, then apply to A in its support
    if np.any(overlap):
        a, b = _lin_gain_offset(A_can, B_can, overlap)
        A_can = a[None, None, :] * A_can + b[None, None, :]

    # Distance-based weights
    wA = _distance_weight(Am.astype(np.uint8))
    wB = _distance_weight(Bm.astype(np.uint8))

    if np.any(overlap):
        # sharpen overlap weights via strength parameter (0→uniform, 1→distance)
        wA_ov = wA * overlap + 1e-6
        wB_ov = wB * overlap + 1e-6
        # blend weights
        w_sum_ov = wA_ov ** (1.0 * strength) + wB_ov ** (1.0 * strength)
        wA_blend = np.where(
            overlap > 0, (wA_ov**strength) / w_sum_ov, Am.astype(np.float32)
        )
        wB_blend = np.where(
            overlap > 0, (wB_ov**strength) / w_sum_ov, Bm.astype(np.float32)
        )
    else:
        # no overlap — simple paste
        wA_blend = Am.astype(np.float32)
        wB_blend = Bm.astype(np.float32)

    # Normalize outside overlap to avoid zeros
    w_sum = wA_blend + wB_blend + 1e-6
    wA_blend /= w_sum
    wB_blend /= w_sum

    fused = A_can * wA_blend[..., None] + B_can * wB_blend[..., None]

    if clamp:
        fused = np.clip(fused, 0.0, 1.0)

    return fused[..., 0]
