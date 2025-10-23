import numpy as np
import cv2 as cv


def _distance_weight(mask: np.ndarray, min_inside: float = 1.0) -> np.ndarray:
    """
    Distance-to-boundary weights; strictly zero outside support.
    mask: uint8 {0,1}
    Returns float32 weights with >= min_inside where mask==1, and 0 where mask==0.
    """
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    # distance to background, defined only on foreground; zeros on background
    dist_in = cv.distanceTransform(m * 255, cv.DIST_L2, 3).astype(np.float32)
    # ensure interior has at least min_inside; keep background exactly zero
    w = np.where(m == 1, dist_in + float(min_inside), 0.0).astype(np.float32)
    return w


def fuse_many_images(
    images,  # List[np.ndarray], HxW or HxWxC
    offsets_xy,  # List[Tuple[float,float]] (dx, dy) per image
    strength: float = 1.0,
    clamp: bool = True,
):
    # -- normalize dtype & channels
    def _to_float(img):
        return (
            (img.astype(np.float32) / 255.0)
            if img.dtype == np.uint8
            else img.astype(np.float32)
        )

    imgs = [_to_float(im) for im in images]
    C = 1 if imgs[0].ndim == 2 else imgs[0].shape[2]
    imgs = [im[..., None] if im.ndim == 2 else im for im in imgs]
    for im in imgs:
        if im.shape[2] != C:
            raise ValueError("All images must have the same number of channels.")

    # -- canvas geometry
    def _compute_canvas_shape_many(images, offsets):
        corners = []
        for im, (dx, dy) in zip(images, offsets):
            H, W = im.shape[:2]
            corners.append((dx, dy))
            corners.append((dx + W, dy + H))
        xs = [x for x, _ in corners]
        ys = [y for _, y in corners]
        x_min, x_max = int(np.floor(min(xs))), int(np.ceil(max(xs)))
        y_min, y_max = int(np.floor(min(ys))), int(np.ceil(max(ys)))
        return (y_max - y_min, x_max - x_min), (-x_min, -y_min)

    (Hc, Wc), shift = _compute_canvas_shape_many(imgs, offsets_xy)
    canvas_shape = (Hc, Wc, C)

    # -- place images & build STRICT masks
    def _place_on_canvas(img, offset, canvas_shape):
        Hc, Wc = canvas_shape[:2]
        H, W = img.shape[:2]
        dx, dy = offset
        x0 = int(round(max(dx, 0)))
        y0 = int(round(max(dy, 0)))
        x1 = min(x0 + W, Wc)
        y1 = min(y0 + H, Hc)
        sx0 = int(round(max(-dx, 0)))
        sy0 = int(round(max(-dy, 0)))
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)
        can = np.zeros(canvas_shape, np.float32)
        m = np.zeros((Hc, Wc), np.uint8)
        if x0 < x1 and y0 < y1:
            can[y0:y1, x0:x1] = img[sy0:sy1, sx0:sx1]
            m[y0:y1, x0:x1] = 1
        return can, m

    placed, masks, weights = [], [], []
    for im, (dx, dy) in zip(imgs, offsets_xy):
        can, m = _place_on_canvas(im, (dx + shift[0], dy + shift[1]), canvas_shape)
        # weights are strictly zero outside support
        w = _distance_weight(m, min_inside=1.0)
        # apply seam “sharpness” via exponent but KEEP background zero
        if strength != 1.0:
            w = np.where(m == 1, np.power(w, max(strength, 0.0)), 0.0).astype(
                np.float32
            )
        placed.append(can)
        masks.append(m)
        weights.append(w)

    # -- precompute totals without leaking background
    S_total = np.zeros(canvas_shape, np.float32)
    W_total = np.zeros((Hc, Wc), np.float32)
    for I, w in zip(placed, weights):
        S_total += I * w[..., None]
        W_total += w  # zero where no support

    # -- per-image exposure compensation against ALL OTHERS on strict overlap
    def _lin_gain_offset(src, ref, mask):
        eps = 1e-8
        m = mask.reshape(-1) > 0
        if not np.any(m):
            return np.ones((src.shape[2],), np.float32), np.zeros(
                (src.shape[2],), np.float32
            )
        src_f = src.reshape(-1, src.shape[2])[m]
        ref_f = ref.reshape(-1, ref.shape[2])[m]
        a = np.empty(src.shape[2], np.float32)
        b = np.empty(src.shape[2], np.float32)
        N = src_f.shape[0]
        for c in range(src.shape[2]):
            x = src_f[:, c]
            y = ref_f[:, c]
            Sx = float(x.sum())
            Sy = float(y.sum())
            Sxx = float((x * x).sum())
            Sxy = float((x * y).sum())
            denom = (Sxx * N - Sx * Sx) + eps
            a[c] = (Sxy * N - Sx * Sy) / denom
            b[c] = (Sy - a[c] * Sx) / (N + eps)
        return a, b

    adjusted = []
    for I_k, M_k, w_k in zip(placed, masks, weights):
        # others-only reference
        S_ref = S_total - (I_k * w_k[..., None])
        W_ref = W_total - w_k
        # overlap: pixels where this image exists AND others exist (strict)
        valid = M_k.astype(bool) & (W_ref > 0)
        if np.any(valid):
            R = np.zeros_like(I_k)
            R[valid, :] = S_ref[valid, :] / (W_ref[valid, None] + 1e-12)
            a, b = _lin_gain_offset(I_k, R, valid.astype(np.uint8))
            I_adj = a[None, None, :] * I_k + b[None, None, :]
        else:
            I_adj = I_k
        adjusted.append(I_adj)

    # -- final composite (no background weight leakage)
    S_adj = np.zeros(canvas_shape, np.float32)
    W_sum = np.zeros((Hc, Wc), np.float32)
    for I_adj, w in zip(adjusted, weights):
        S_adj += I_adj * w[..., None]
        W_sum += w

    fused = S_adj / np.maximum(W_sum[..., None], 1e-12)
    if clamp:
        fused = np.clip(fused, 0.0, 1.0)

    return fused[..., 0] if C == 1 else fused
